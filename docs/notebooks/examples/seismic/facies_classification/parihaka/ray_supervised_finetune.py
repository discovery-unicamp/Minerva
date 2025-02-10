import argparse
import ray

from finetune_dinov2_dpt import train as train_dinov2_dpt
from finetune_dinov2_mla import train as train_dinov2_mla
from finetune_dinov2_pup import train as train_dinov2_pup
from finetune_fastsiam import train as train_fastsiam
from finetune_kenshodense import train as train_kenshodense
from finetune_sfm_base_patch16 import train as train_sfm_base_patch16
from finetune_simclr import train as train_simclr
from finetune_tribyol import train as train_tribyol
from supervised_deeplabv3 import train as train_deeplabv3
from finetune_byol import train as train_byol
from finetune_lfr import train as train_lfr
from finetune_sam import train as train_sam
from supervised_deeplabv3 import train as train_deeplabv3
from finetune_setr_pup import train as train_setr_pup


def main_finetune_ray():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ray-address",
        type=str,
        default="auto",
        help="Address of Ray cluster for distributed training.",
    )

    args = parser.parse_args()
    ray.init(address=args.ray_address)

    remotes = []
    for func in [
        train_dinov2_dpt,
        train_dinov2_mla,
        train_dinov2_pup,
        train_fastsiam,
        train_kenshodense,
        train_sfm_base_patch16,
        train_simclr,
        train_tribyol,
        train_byol,
        train_lfr,
        train_deeplabv3,
        train_sam,
        # train_setr_pup
    ]:
        name = f"{func.__module__}.{func.__name__}"
        remote_func = ray.remote(max_retries=1, num_gpus=1, num_cpus=8)(func)
        remote_func = remote_func.options(name=name)
        future = remote_func.remote()
        remotes.append(future)
        
        print(f"Started {name}. Remote: {future}")

    ray.get(remotes)


if __name__ == "__main__":
    main_finetune_ray()
