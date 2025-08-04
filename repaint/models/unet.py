from abc import abstractmethod
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .unet_utils import (
    conv_transpose_nd, conv_nd, linear, avg_pool_nd,
    zero_module, normalization,
    CheckpointFunction, timestep_embedding,
    convert_module_to_f16, convert_module_to_f32,
)


class TimestepBlock(nn.Module):
    """继承 nn.Module, 并将 forward 方法变成一个抽象方法(只声明不实现)"""

    @abstractmethod
    def forward(self, x, emb):
        """让其强制转变为 x 和 emb 两个参数"""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """一个顺序模块，将时间步嵌入作为额外输入传递给支持它的子模块"""

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """ 上采样, 直接使用 ConvTranspose2d """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
            
    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """ 下采样, 可选方式：卷积网络层、平均池化网络层 """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """ 残差块关键词: 时间步嵌入方式、模块参数清零、梯度检查点、上下采样集成 """

    def __init__(
        self, channels, emb_channels, dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        # --------------- 是否使用梯度检查点 ---------------
        if self.use_checkpoint:
            inputs = (x, emb)
            args = tuple(inputs) + tuple(self.parameters())
            return CheckpointFunction.apply(self._forward, len(inputs), *args)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
        # 集成上下采样，主类中不必再写分支逻辑处理尺寸
        if self.updown:
            # 先进行归一化、SiLU激活，进行上下采样，最后走卷积网络层
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)  # 残差路径上下采样
            x = self.x_upd(x)  # 主路径上下采样
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # --------------- 选择时间步嵌入的方式 ---------------
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """ 允许空间位置相互注意 """

    def __init__(
        self, channels, num_heads=1, num_head_channels=-1,
        use_checkpoint=False, use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (channels % num_head_channels == 0), \
                f"q,k,v channels is not divisible by num_head_channels"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        # 前者是先分割 qkv 再分割 heads，后者相反
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        if self.use_checkpoint:
            inputs = (x,)
            args = tuple(inputs) + tuple(self.parameters())
            return CheckpointFunction.apply(self._forward, len(inputs), *args)
        else:
            return self._forward(x)

    def _forward(self, x):
        # 保存原始的 B、C、H、W，然后将图像 x 的空间形状压缩为一维 [b, c, hw]
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        # Predict core key values using a 1x1 convolusion (h*w -> 3*h*2)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)

        # 输出映射，残差连接 + 恢复原有的图像结构
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """ 先分割 heads, 再分 qkv; reshape 容易出错，但推理和训练效率略快、适合 CUDA 加速 """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        # bs = batch_size，width = 3C 是通道维度，length = T = H × W
        # (H * 3 * C) 中的 H 代表 heads，3 是卷积之后通道数扩大 3 倍，C为原始的通道数
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        # 先分好 heads，再用 split 将通道维度分割成 qkv
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3,length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        pass


class QKVAttention(nn.Module):
    """ 先分割 qkv, 再分 heads; 语义清晰，但内存访问连续性一般、推理和训练效率稍慢 """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        
        q, k, v = qkv.chunk(3, dim=1)  # 先切割 QKV，沿着通道维度
        scale = 1 / math.sqrt(math.sqrt(ch))  # 用于数值稳定的缩放
        # 切割 q 和 k 的 heads，使用 einsum 进行矩阵乘法
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        # 使用 .float() 防止 FP16 underflow，进行 softmax，最后数值类型再转回去
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 加权求值，V × 注意力；reshape 回原始形式并返回
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        pass


class UNetModel(nn.Module):
    """ The full UNet model with attention and timestep embedding. """

    def __init__(
        self, image_size, in_channels, model_channels, out_channels,
        num_res_blocks, attention_resolutions, dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        conf=None
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.conf = conf

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level != 0 and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch,
                            dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm, up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """ Convert the torso of the model to float16. """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """ Convert the torso of the model to float32. """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, gt=None, **kwargs):
        """ Apply the model to an input batch. """

        # if timesteps[0].item() > self.conf.diffusion.steps:
        #     raise RuntimeError("timesteps larger than diffusion steps.",
        #                        timesteps[0].item(), self.conf.diffusion_steps)

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(UNetModel):
    """
    执行超分辨率的 UNetModel。
    希望有一个额外的 kwarg `low_res` 来作为低分辨率图像的条件。
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)
