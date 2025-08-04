""" Train a diffusion model on images. """
import torch as th
from tools import logger
from tools.image_datasets2 import load_data
from models.resample import create_named_schedule_sampler
from tools.script_util import create_model_and_diffusion, load_config
from tools.train_util import TrainLoop


def main():
    # ------------ 参数字典、硬件设备、日志文件的初始化 ------------
    config = load_config("repaint/configs/test_256.yml")
    args_t = config['training']
    args_m = config['model']
    args_d = config['diffusion']
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print("device: ", device)
    logger.configure()
        
    # ------------ 扩散模型、神经网络、重要性采样的初始化 ------------
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args_m, args_d)
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args_t["schedule_sampler"], diffusion)

    # ------------ 数据集图像的预处理 ------------
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args_t["data_dir"],
        batch_size=args_t["batch_size"],
        image_size=args_m["image_size"],
        class_cond=args_m["class_cond"],
    )

    # ------------ 开始走训练流程 ------------
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args_t["batch_size"],
        microbatch=args_t["microbatch"],
        lr=args_t["lr"],
        ema_rate=args_t["ema_rate"],
        log_interval=args_t["log_interval"],
        save_interval=args_t["save_interval"],
        resume_checkpoint=args_t["resume_checkpoint"],
        use_fp16=args_t["use_fp16"],
        fp16_scale_growth=args_t["fp16_scale_growth"],
        schedule_sampler=schedule_sampler,
        weight_decay=args_t["weight_decay"],
        lr_anneal_steps=args_t["lr_anneal_steps"],
    ).run_loop()


if __name__ == "__main__":
    main()
