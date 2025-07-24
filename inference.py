# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import functools
import json

import jax
import jax.numpy as jnp
import tensorflow as tf
from flax.training import checkpoints as flax_checkpoints

from vit_jax import models
from vit_jax.configs import models as config_lib
from vit_jax.preprocess import PreprocessImages

# 1. 配置参数
image_size = 384  # 根据您的模型配置
num_classes = 1000  # 根据您的分类任务
checkpoint_dir = "/home/wumianzi/workspace/codespace/vision_transformer/vit_train/checkpoint_20000"  # 修改为您的检查点路径
model_name = "ViT-B_16"  # 例如: "ViT-B_16", "ViT-L_32" 等

# 正确方式: 使用模型配置字典而不是集合
model_config = config_lib.MODEL_CONFIGS[model_name]
# 或者手动定义一个配置字典:
# model_config = {
#     'hidden_size': 768,
#     'patches': {'size': 16},
#     'transformer': {'mlp_dim': 3072, 'num_heads': 12, 'num_layers': 12, 'attention_dropout_rate': 0.0, 'dropout_rate': 0.1},
#     'representation_size': None,
# }

# 2. 初始化模型
model = models.VisionTransformer(num_classes=num_classes, **model_config)


# 3. 创建初始参数
@functools.partial(jax.jit, backend='cpu')
def init_model():
    return model.init(jax.random.PRNGKey(0), jnp.ones([1, image_size, image_size, 3], jnp.float32), train=False)


variables = init_model()
params = variables['params']

# 4. 加载检查点
params, opt_state, step = flax_checkpoints.restore_checkpoint(checkpoint_dir, (params, None, 0))

# 5. 创建推理函数
infer_fn = jax.jit(functools.partial(model.apply, train=False))

# 6. 创建预处理器
preprocessor = PreprocessImages(size=image_size, crop=False)


# 7. 加载和处理图像
def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, 3)
    processed = preprocessor.preprocess_tf(image).numpy()
    return processed


# 8. 推理函数
def predict(image_path, class_names=None):
    # 处理图像
    img = process_image(image_path)
    img = img[None, ...]  # 添加批次维度

    # 推理
    logits = infer_fn({'params': params}, img)
    probabilities = jax.nn.softmax(logits, axis=-1).squeeze(0)
    top5_classes = jnp.argsort(probabilities)[-5:][::-1]
    top5_probabilities = probabilities[top5_classes]

    # 映射
    class_label_map = json.load(open("/home/wumianzi/workspace/ImageNet_1K/ImageNet_1K_labels_map.txt"))
    class_label_map = {int(key): value for key, value in class_label_map.items()}

    print()
    for i in range(5):
        prediction_class_label = class_label_map[top5_classes.tolist()[i]]
        prediction_class_prob = top5_probabilities[i]
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using Vision Transformer")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    predict(args.image_path)