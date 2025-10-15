import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm # 用于显示进度条
import re
import time

from mobile_cv.model_zoo.models.fbnet_tf_a import FBNetAKeras
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet

GROUND_TRUTH_PATH = '/root/autodl-tmp/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
# --- 配置参数 ---
TFLITE_MODEL_PATH = '/root/autodl-tmp/deploy/fbnet_a_imagenet_int8_224x224_model.tflite' # 替换为你的量化模型路径
NUM_TEST_SAMPLES = 1000  # 用于精度评估的样本数量
INPUT_SIZE = 224
IMAGENET_VAL_DIR = '/root/autodl-tmp/imagenet-1k' # 替换为你的 ImageNet 验证集目录

keras_model_path = '/root/autodl-tmp/deploy/fbnet_a.keras'
keras_model = tf.keras.models.load_model(
    keras_model_path,
    custom_objects={'FBNetAKeras': FBNetAKeras} # 将类名映射到实际的 Python 类
)
input_shape_without_batch = (224, 224, 3) 
keras_model.build(input_shape=(None,) + input_shape_without_batch)

# PyTorch Normalize 参数
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])
INPUT_CROP_SIZE = 224

def load_imagenet_ground_truth(file_path):
    """从文件中加载 ImageNet 验证集的所有真实标签。"""
    print(f"正在加载真实标签文件: {file_path}")
    with open(file_path, 'r') as f:
        # 标签文件是 1-based 索引
        # 我们将它存储在一个列表中，索引 0 对应第一张图 (0-based)
        labels = [int(line.strip()) for line in f if line.strip()]
    return labels


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

def load_test_data(num_samples):
    """
    加载 ImageNet 测试图像及其真实标签。
    要求图像文件名为 ILSVRC2012_val_00000001.JPEG 格式。
    """
    # 1. 加载所有真实标签
    try:
        all_labels_1_based = load_imagenet_ground_truth(GROUND_TRUTH_PATH)
    except FileNotFoundError:
        print(f"错误: 标签文件未找到在 {GROUND_TRUTH_PATH}。无法进行精度评估。")
        return []

    image_files = os.listdir(IMAGENET_VAL_DIR)
    
    # 过滤掉非 JPEG 文件
    image_files = [f for f in image_files if f.endswith(('.JPEG', '.jpg', '.jpeg'))]

    # 随机选择样本
    # np.random.shuffle(image_files) # 如果需要随机性，请解除注释
    selected_files = image_files[:num_samples]
    
    test_data = [] 
    
    print(f"正在加载并预处理 {len(selected_files)} 个图像样本...")

    # 正则表达式用于从文件名中提取序号
    # ILSVRC2012_val_00000001.JPEG -> 1
    file_pattern = re.compile(r'ILSVRC2012_val_(\d+)\.JPEG', re.IGNORECASE)

    for filename in selected_files:
        match = file_pattern.search(filename)
        
        if match:
            # 提取序号 (例如 '00000001' -> 1)
            file_index_1_based = int(match.group(1)) 
            # 标签列表是 0-based，所以索引需要减 1
            label_index_0_based = file_index_1_based - 1
            
            # 检查索引是否有效
            if 0 <= label_index_0_based < len(all_labels_1_based):
                true_label = all_labels_1_based[label_index_0_based]
                
                image_path = os.path.join(IMAGENET_VAL_DIR, filename)
                
                # 预处理
                processed_image = tf_fbnet_preprocess(image_path)
                processed_image = tf.expand_dims(processed_image, axis=0) 
                
                # 存储 [输入张量, 真实标签]
                test_data.append((processed_image.numpy(), true_label))
            else:
                print(f"警告: 文件 {filename} 的索引 {label_index_0_based} 超出标签范围。跳过。")
        else:
            print(f"警告: 文件名 {filename} 不符合标准 ImageNet 验证集格式。跳过。")
    print(f'数据集长度{len(test_data)}')
    return test_data
# =================================================================
# 2. TFLite Interpreter 推理函数
# =================================================================

def run_tflite_inference(tflite_model_path, test_data):
    # 初始化 TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # 获取输入和输出张量的索引
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # TFLite 推理循环
    tflite_outputs = []
    
    # 获取 TFLite 模型的期望输入类型
    # 如果全整数量化成功，这里应该是 tf.int8 或 tf.uint8
    input_dtype = input_details['dtype']
    
    print(f"\n📢 TFLite 模型期望输入类型: {input_dtype}")

    start_time = time.time()

    for input_tensor, label in tqdm(test_data, desc="TFLite 推理中"):
        # 1. 量化输入（如果 TFLite 模型要求 INT8 输入）
        if input_dtype == np.int8 or input_dtype == np.uint8:
            # 获取量化参数
            scale, zero_point = input_details['quantization']
            try:
                info = np.iinfo(input_dtype)
                input_min_value = info.min
                input_max_value = info.max
            except ValueError:
                # 如果是 float32 或其他非整数类型，则直接使用 np.finfo
                info = np.finfo(input_dtype)
                input_min_value = info.min
                input_max_value = info.max
            # 将 Float32 输入量化为 INT8
            if scale == 0:
                quantized_input = input_tensor.astype(input_dtype)
            else:
                quantized_input = input_tensor / scale + zero_point
                quantized_input = np.clip(quantized_input, input_min_value, input_max_value).astype(input_dtype)
        else:
            # 如果 TFLite 仍然要求 Float32 输入 (动态范围量化)
            quantized_input = input_tensor

        
        # 2. 设置输入
        interpreter.set_tensor(input_details['index'], quantized_input)
        
        # 3. 执行推理
        interpreter.invoke()
        
        # 4. 获取输出
        tflite_output = interpreter.get_tensor(output_details['index'])
        
        # 5. 反量化输出（如果输出是 INT8）
        if output_details['dtype'] == np.int8 or output_details['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details['quantization']
            if output_scale != 0:
                tflite_output = (tflite_output.astype(np.float32) - output_zero_point) * output_scale
        
        tflite_outputs.append(tflite_output)

    end_time = time.time()
    
    total_time = end_time - start_time
    fps = len(test_data) / total_time
    
    print(f"TFLite 推理完成。总耗时: {total_time:.4f} 秒")
    print(f"TFLite 吞吐量 (FPS): {fps:.2f} 样本/秒")
        
    return np.array(tflite_outputs)

# =================================================================
# 3. Keras 模型推理函数 (作为 Float32 基准)
# =================================================================

def run_keras_inference(keras_model, test_data):
    keras_inputs = [data[0] for data in test_data]
    
    # Keras 推理通常比 TFLite Interpreter 快
    print("\n🚀 Keras (Float32) 推理中...")

    # keras_outputs = []
    # total_inference_time = 0
    # for input_tensor, label in tqdm(test_data, desc="Keras 单样本推理中"):
    #     start_time = time.time()
    #     # 强制 batch_size=1
    #     output = keras_model.predict(input_tensor, batch_size=1, verbose=0) 
    #     end_time = time.time()
    #     total_inference_time += (end_time - start_time)
    #     keras_outputs.append(output)

    # keras_outputs = np.array(keras_outputs)

    # 注意：这里的 keras_model 应该被禁用 GPU 以便公平比较，但为了简单，直接运行
    start_time = time.time()
    keras_outputs = keras_model.predict(np.vstack(keras_inputs), batch_size=32, verbose=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = len(test_data) / total_time
    
    print(f"Keras 推理完成。总耗时: {total_time:.4f} 秒")
    print(f"Keras 吞吐量 (FPS): {fps:.2f} 样本/秒")
    
    return keras_outputs

# =================================================================
# 4. 精度评估与报告
# =================================================================

def evaluate_and_report(tflite_outputs, keras_outputs, test_data):
    # 获取真实标签
    true_labels = np.array([data[1] for data in test_data])
    
    # 预测类别
    tflite_preds = np.argmax(tflite_outputs, axis=1)
    keras_preds = np.argmax(keras_outputs, axis=1)
    
    # 1. 计算 Top-1 精度 (需要真实标签)
    tflite_accuracy = np.mean(tflite_preds == true_labels)
    keras_accuracy = np.mean(keras_preds == true_labels)

    # 2. 计算输出差异 (量化损失的直接指标，不需要真实标签)
    # 使用 L2 范数或余弦相似度比较 Float32 和 INT8 的输出 logits
    output_difference = np.linalg.norm(tflite_outputs - keras_outputs, axis=1).mean()

    # 3. 预测一致性 (两个模型预测结果相同的比例)
    consistency = np.mean(tflite_preds == keras_preds)
    
    print("\n--- 精度验证报告 ---")
    print(f"样本总数: {len(test_data)}")
    print(f"1. 预测一致性 (INT8 vs Float32): {consistency:.4f} ({consistency*100:.2f}%)")
    print(f"2. 平均输出 L2 差异 (越小越好): {output_difference:.6f}")
    
    if np.any(true_labels != 0): # 只有当标签是真实值时才计算精度
        print(f"3. Keras (Float32) Top-1 精度: {keras_accuracy:.4f}")
        print(f"4. TFLite (INT8) Top-1 精度: {tflite_accuracy:.4f}")
        print(f"量化损失 (Float32 - INT8): {(keras_accuracy - tflite_accuracy)*100:.2f}%")
    else:
        print("注意: 由于缺乏真实标签，无法计算 Top-1 精度。")

# =================================================================
# 5. 主执行逻辑
# =================================================================

if __name__ == '__main__':
    # 1. 加载测试数据
    test_data = load_test_data(NUM_TEST_SAMPLES)
    if not test_data:
        print("错误: 未加载到任何测试数据。请检查 IMAGENET_VAL_DIR 路径。")
    else:
        # 2. Keras 推理 (Float32 基准)
        keras_outputs = run_keras_inference(keras_model, test_data)
        
        # 3. TFLite Interpreter 推理 (量化模型)
        tflite_outputs = run_tflite_inference(TFLITE_MODEL_PATH, test_data)
        
        # 4. 报告结果
        evaluate_and_report(tflite_outputs, keras_outputs, test_data)
        
    print("\n✅ 量化模型验证流程结束。")