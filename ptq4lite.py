import tensorflow as tf
import numpy as np
import os
import random # 用于随机选择样本

from mobile_cv.model_zoo.models.fbnet_tf_a import FBNetAKeras
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet

keras_model_path = '/root/autodl-tmp/deploy/fbnet_a.keras'
keras_model = tf.keras.models.load_model(
    keras_model_path,
    custom_objects={'FBNetAKeras': FBNetAKeras} # 将类名映射到实际的 Python 类
)
input_shape_without_batch = (224, 224, 3) 
keras_model.build(input_shape=(None,) + input_shape_without_batch)

# 2. ImageNet 验证集图像所在的目录
IMAGENET_VAL_DIR = '/root/autodl-tmp/imagenet-1k'
NUM_CALIBRATION_SAMPLES = 500 # 建议 500 个样本
INPUT_SIZE = 224

# =================================================================
# 1. ImageNet 数据加载和预处理 (核心部分)
# =================================================================

# PyTorch Normalize 参数
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])
INPUT_CROP_SIZE = 224

def tf_fbnet_preprocess(image_path):
    """
    TensorFlow 实现 FBNet 官方仓库的 PyTorch 预处理逻辑。
    假设目标 resize 尺寸是 256。
    """
    
    # --- 1. 读取图像 ---
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) # 原始 uint8 [0, 255]

    # --- 2. 缩放最短边到 256 ---
    # TensorFlow 推荐使用 tf.image.resize 和 tf.image.resize_with_pad 实现，
    # 但最精确的方法是手动计算宽高比。
    original_shape = tf.shape(img)
    orig_h = tf.cast(original_shape[0], tf.float32)
    orig_w = tf.cast(original_shape[1], tf.float32)
    
    # 确定缩放因子：缩放最短边到 256
    scale_factor = tf.cond(orig_h < orig_w, 
                           lambda: 256.0 / orig_h, 
                           lambda: 256.0 / orig_w)
    
    new_h = tf.cast(orig_h * scale_factor, tf.int32)
    new_w = tf.cast(orig_w * scale_factor, tf.int32)
    
    # 执行缩放
    resized_img = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)

    # --- 3. 中心裁剪到 224x224 ---
    cropped_img = tf.image.central_crop(resized_img, central_fraction=INPUT_CROP_SIZE / tf.cast(tf.minimum(new_h, new_w), tf.float32))

    # 由于 tf.image.central_crop 接受 fraction，使用 tf.image.crop_to_bounding_box 更精确控制尺寸
    # 重新计算裁剪起始点
    h_start = (new_h - INPUT_CROP_SIZE) // 2
    w_start = (new_w - INPUT_CROP_SIZE) // 2
    cropped_img = tf.image.crop_to_bounding_box(resized_img, h_start, w_start, INPUT_CROP_SIZE, INPUT_CROP_SIZE)

    # --- 4. 转换为 [0, 1] 的 Float32 ---
    # PyTorch ToTensor() 行为
    normalized_img = tf.image.convert_image_dtype(cropped_img, tf.float32) 

    # --- 5. 标准化 (Normalize) ---
    # PyTorch: CHW format -> TF: HWC format, 需要扩展维度并重新排序，但 tf.image 已经在 HWC 上操作
    # mean 和 std 需要扩展到 HWC 的形状进行广播
    mean_tensor = tf.reshape(IMAGENET_MEAN, [1, 1, 3])
    std_tensor = tf.reshape(IMAGENET_STD, [1, 1, 3])
    
    # img = (img - mean) / std
    final_img = (normalized_img - mean_tensor) / std_tensor
    
    # 返回 HWC 格式的最终张量
    return final_img


def representative_data_gen():
    """
    用于 TFLite 转换器的生成器函数。
    它加载、预处理图像，并以 Batch Size 1 的格式 yield 张量。
    """
    # 确保路径存在
    if not os.path.isdir(IMAGENET_VAL_DIR):
        raise FileNotFoundError(f"ImageNet 验证集目录不存在: {IMAGENET_VAL_DIR}")
        
    image_files = [f for f in os.listdir(IMAGENET_VAL_DIR) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    # 随机选择样本以提高代表性
    random.shuffle(image_files)
    selected_files = image_files[:NUM_CALIBRATION_SAMPLES]
    
    print(f"将使用 {len(selected_files)} 个样本进行 PTQ 校准...")
    
    for filename in selected_files:
        image_path = os.path.join(IMAGENET_VAL_DIR, filename)
        
        # 预处理
        processed_image = tf_fbnet_preprocess(image_path)
        
        # 添加 Batch 维度 (1, 224, 224, 3)
        processed_image = tf.expand_dims(processed_image, axis=0)
        
        # 必须 yield [输入张量]，且类型为 tf.float32
        yield [processed_image]

# =================================================================
# 2. TFLite 转换和量化
# =================================================================

def convert_to_tflite_int8(keras_model, output_path):
    """执行全整数量化转换"""
    
    # 1. 实例化转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # 2. 开启默认优化 (PTQ 的第一步)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 3. 设置代表性数据集
    converter.representative_dataset = representative_data_gen
    
    # 4. **强制全整数量化 (8-bit 权重 + 8-bit 激活)**
    # 这是关键，将所有浮点操作都映射到 INT8 操作
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 5. 设置输入和输出张量类型为 INT8 (推荐用于性能)
    # 转换器会在输入/输出自动添加量化/反量化节点
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # 6. 执行转换
    try:
        tflite_quant_model = converter.convert()
        
        # 7. 保存模型
        with open(output_path, 'wb') as f:
            f.write(tflite_quant_model)
            
        print(f"\n✅ 成功！全整数量化模型已保存到: {output_path}")
        print(f"模型大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return tflite_quant_model
        
    except Exception as e:
        print(f"\n❌ 警告/错误: 转换失败。可能是某些层不支持 INT8。")
        print(f"详细错误: {e}")
        return None

if __name__ == '__main__':
    OUTPUT_TFLITE_PATH = 'fbnet_a_imagenet_int8_224x224_model.tflite'
    
    # print(f"开始使用 {NUM_CALIBRATION_SAMPLES} 个样本进行 PTQ 转换...")
    
    # 执行转换
    converted_model = convert_to_tflite_int8(keras_model, OUTPUT_TFLITE_PATH)

    # 提示用户下一步可以验证模型精度
    if converted_model:
        print("\n下一步：请使用 TFLite Interpreter 验证量化模型的精度，确保量化损失在可接受范围内。")
