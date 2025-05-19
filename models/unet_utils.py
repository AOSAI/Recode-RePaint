import math
import torch as th
import torch.nn as nn


# --------------- 网络层组件 ---------------
def conv_transpose_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D transposed convolution module."""
    if dims == 1:
        return nn.ConvTranspose1d(*args, **kwargs)
    elif dims == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    elif dims == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


# --------------- 组归一化 - 重写 ---------------
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def normalization(channels):
    """调用重构后的 GroupNorm32"""
    return GroupNorm32(32, channels)


# --------------- 清零模块参数 ---------------
def zero_module(module):
    """仅在模型结构初始化的时候调用一次！"""
    for p in module.parameters():
        p.detach().zero_()
    return module


# --------------- 梯度检查点机制 ---------------
class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
    

# --------------- 时间步的正余弦高频编码 ---------------
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: LongTensor, 形状为 [batch_size], 表示每张图像的时间步t
    :param dim: int, 时间嵌入的维度 (通常等于 base_channels * 4)
    :param max_period: 控制嵌入的最小频率
    :return: Tensor: [batch_size, dim] 的时间嵌入向量
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# --------------- 精度转换，原本写在 fp16_util 文件中 ---------------
def convert_module_to_f16(l):
    """ Convert primitive modules to float16. """
    moduleType = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, 
        nn.ConvTranspose3d
    )
    if isinstance(l, moduleType):
        l.weight.data = l.weight.data.half()
        l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """ Convert primitive modules to float32. """
    moduleType = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, 
        nn.ConvTranspose3d
    )
    if isinstance(l, moduleType):
        l.weight.data = l.weight.data.float()
        l.bias.data = l.bias.data.float()


