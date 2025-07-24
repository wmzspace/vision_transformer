# ViT

## Train

### Big 16

```shell
CUDA_VISIBLE_DEVICES=0,1,4,6 LD_PRELOAD=/home/wumianzi/miniconda3/envs/ViT/lib/libcudart.so.12:/home/wumianzi/miniconda3/envs/ViT/lib/libnccl.so python -m vit_jax.main --workdir=/home/wumianzi/workspace/codespace/vision_transformer/vit_train \
    --config=$(pwd)/vit_jax/configs/vit.py:b16,imagenet2012 \
    --config.pretrained_dir='/home/wumianzi/workspace/codespace/vision_transformer/vit_models/imagenet21k' \
    --config.dataset='/home/wumianzi/workspace/ImageNet_1K' \
    --config.batch=256
```

### TIny 16

```shell
CUDA_VISIBLE_DEVICES=0,1,4,6 LD_PRELOAD=/home/wumianzi/miniconda3/envs/ViT/lib/libcudart.so.12:/home/wumianzi/miniconda3/envs/ViT/lib/libnccl.so python -m vit_jax.main --workdir=/home/wumianzi/workspace/codespace/vision_transformer/vit_train \
    --config=$(pwd)/vit_jax/configs/augreg.py:Ti_16 \
    --config.dataset=/home/wumianzi/workspace/ImageNet_1K \
    --config.pretrained_dir=/home/wumianzi/workspace/codespace/vision_transformer/vit_models/imagenet21k/augreg \
    --config.base_lr=0.01 \
    --config.batch=256 \
    --config.shuffle_buffer=50000
```

## Environment

```shell
CUDA_VISIBLE_DEVICES=6,7 LD_PRELOAD=/home/wumianzi/miniconda3/envs/ViT/lib/libcudart.so.12:/home/wumianzi/miniconda3/envs/ViT/lib/libnccl.so python inference.py
```

## Inference

```shell
python inference.py tinca.JPEG
```