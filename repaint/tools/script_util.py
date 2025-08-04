import yaml
from models.respace import SpacedDiffusion, space_timesteps
from models.unet import UNetModel
from models.unet2 import EncoderUNetModel
from models.noise_schedule import get_noise_schedule
from models.diff_utils import ModelMeanType, ModelVarType, LossType

NUM_CLASSES = 1000

def create_model_and_diffusion(model_kwargs, diffusion_kwargs):
    model = create_model(**model_kwargs)
    diffusion = create_gaussian_diffusion(**diffusion_kwargs)
    return model, diffusion

# 通过图像尺寸得到 channel_mult 的最优解（base_channel为128）
def get_channel_mult(channel_mult, image_size):
    presets = {
        512: (0.5, 1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 2),
    }
    if channel_mult == "":
        if image_size not in presets:
            raise ValueError(f"Unsupported image size: {image_size}")
        return presets[image_size]
    else:
        if isinstance(channel_mult, tuple):
            return channel_mult
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
        return channel_mult

def create_model(
    image_size, num_channels, num_res_blocks, channel_mult="",
    learn_sigma=False, class_cond=False, use_checkpoint=False,
    attention_resolutions="16", num_heads=1,
    num_head_channels=-1, num_heads_upsample=-1,
    use_scale_shift_norm=False, dropout=0,
    resblock_updown=False, use_fp16=False,
    use_new_attention_order=False
):
    channel_mult = get_channel_mult(channel_mult, image_size)

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_classifier_model(
    image_size, classifier_use_fp16, classifier_width, classifier_depth,
    classifier_attention_resolutions, classifier_use_scale_shift_norm,
    classifier_resblock_updown, classifier_pool, image_size_inference=None
):
    channel_mult = get_channel_mult(channel_mult, image_size)

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    image_size_inference = image_size_inference or image_size

    return EncoderUNetModel(
        image_size=image_size_inference,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


def create_gaussian_diffusion(
    *, steps=1000, learn_sigma=False, sigma_small=False, noise_schedule="linear", 
    use_kl=False, predict_xstart=False, rescale_timesteps=False, 
    rescale_learned_sigmas=False, timestep_respacing="",
):

    betas = get_noise_schedule(noise_schedule, steps, use_scale=True)

    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE

    if not predict_xstart:
        mean_type = ModelMeanType.EPSILON
    else:
        mean_type = ModelMeanType.START_X

    if not learn_sigma:
        if not sigma_small:
            var_type = ModelVarType.FIXED_LARGE
        else:
            var_type = ModelVarType.FIXED_SMALL
    else:
        var_type = ModelVarType.LEARNED_RANGE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=mean_type,
        model_var_type=var_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        # conf=conf
    )

# 从 ymal 文件中获取参数字典
def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config