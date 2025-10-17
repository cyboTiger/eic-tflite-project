import tensorflow as tf
import numpy as np
import torch
import os
from tqdm import tqdm # 用于显示进度条
import re
import time
from PIL import Image

from mobile_cv.model_zoo.models.fbnet_tf_a import FBNetAKeras
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from mobile_cv.model_zoo.models.preprocess import get_preprocess

GROUND_TRUTH_PATH = '/root/autodl-tmp/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
# --- 配置参数 ---
Q_TFLITE_MODEL_PATH = '/root/autodl-tmp/deploy/fbnet_a_imagenet_int8_224x224_model.tflite' # 替换为你的量化模型路径
TFLITE_MODEL_PATH = '/root/autodl-tmp/deploy/fbnet_a.tflite' # 替换为你的量化模型路径
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

def numpy_softmax(x):
    """
    对 logits (形状为 (N, K)) 进行数值稳定的 Softmax 运算。
    Softmax 沿着最后一个轴 (axis=-1) 计算，将每个样本的 K 个 logit 转换为概率。
    
    参数:
        x (np.ndarray): 输入的 logit 数组，形状为 (num_samples, num_classes)。
        
    返回:
        np.ndarray: 形状与 x 相同的概率数组。
    """
    # 1. 减去每个样本的最大值以保证数值稳定（防止溢出）
    # keepdims=True 保持维度，以便进行广播运算 (N, 1) - (N, K)
    x_max = np.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max
    
    # 2. 计算指数
    e_x = np.exp(x_shifted)
    
    # 3. 计算分母，即指数之和
    # keepdims=True 保持维度，以便进行广播运算 (N, K) / (N, 1)
    e_x_sum = np.sum(e_x, axis=-1, keepdims=True)
    
    # 4. 计算 Softmax 概率
    probabilities = e_x / e_x_sum
    
    return probabilities

def load_imagenet_ground_truth(file_path):
    """从文件中加载 ImageNet 验证集的所有真实标签。"""
    print(f"正在加载真实标签文件: {file_path}")
    with open(file_path, 'r') as f:
        # 标签文件是 1-based 索引
        # 我们将它存储在一个列表中，索引 0 对应第一张图 (0-based)
        labels = [int(line.strip()) for line in f if line.strip()]
    return labels

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
    image_files = sorted(image_files)

    # 随机选择样本
    # np.random.shuffle(image_files) # 如果需要随机性，请解除注释
    selected_files = image_files[:num_samples]
    
    preprocess = get_preprocess(INPUT_CROP_SIZE)
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
                image_data = Image.open(image_path)
                if image_data.mode == 'L':
                    continue
                processed_image = preprocess(image_data)
                
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

    tflite_outputs = np.array(tflite_outputs).squeeze()
        
    return numpy_softmax(tflite_outputs)

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
    
    return numpy_softmax(keras_outputs)

def run_torch_inference(torch_model, test_data):
    test_data = [data[0] for data in test_data]
    test_data = torch.tensor(test_data).squeeze()
    
    start_time = time.time()
    output = torch_model(test_data)
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = len(test_data) / total_time

    print(f"PyTorch 推理完成。总耗时: {total_time:.4f} 秒")
    print(f"PyTorch 吞吐量 (FPS): {fps:.2f} 样本/秒")

    output_softmax = torch.nn.functional.softmax(output, dim=-1).detach().numpy()
    return output_softmax

# =================================================================
# 4. 精度评估与报告
# =================================================================

def evaluate_and_report(tflite_outputs, keras_outputs, torch_outputs, quant_tflite_outputs, test_data):
    # 获取真实标签
    true_labels = np.array([data[1] for data in test_data])
    
    # 预测类别
    tflite_preds = np.argmax(tflite_outputs, axis=1)
    keras_preds = np.argmax(keras_outputs, axis=1)
    torch_preds = np.argmax(torch_outputs, axis=1)
    quant_tflite_preds = np.argmax(quant_tflite_outputs, axis=1)
    
    # 1. 计算 Top-1 精度 (需要真实标签)
    tflite_accuracy = np.mean(tflite_preds == true_labels)
    keras_accuracy = np.mean(keras_preds == true_labels)
    torch_accuracy = np.mean(torch_preds == true_labels)
    quant_tflite_accuracy = np.mean(quant_tflite_preds == true_labels)


    # 2. 计算输出差异 (量化损失的直接指标，不需要真实标签)
    # 使用 L2 范数或余弦相似度比较 Float32 和 INT8 的输出 logits
    output_difference = np.linalg.norm(tflite_outputs - keras_outputs, axis=1).mean()
    quant_output_difference = np.linalg.norm(quant_tflite_outputs - keras_outputs, axis=1).mean()

    # 3. 预测一致性 (两个模型预测结果相同的比例)
    consistency = np.mean(tflite_preds == keras_preds)
    quant_consistency = np.mean(quant_tflite_preds == keras_preds)
    
    print("\n--- 精度验证报告 ---")
    print(f"样本总数: {len(test_data)}")
    print(f"1. 预测一致性 (INT8 TFLite vs Float32 Keras): ({quant_consistency*100:.2f}%)\n (Float32 TFLite vs Float32 Keras): ({consistency*100:.2f}%)\n")
    print(f"2. 平均输出 L2 差异 (越小越好): {output_difference:.6f}(tflite)/{quant_output_difference:.6f}(quantized_tflite)")
    
    if np.any(true_labels != 0): # 只有当标签是真实值时才计算精度
        print(f"3. Keras (Float32) Top-1 精度: {keras_accuracy:.4f}")
        print(f"4. TFLite (INT8) Top-1 精度: {quant_tflite_accuracy:.4f}")
        print(f"4. TFLite (Float32) Top-1 精度: {tflite_accuracy:.4f}")
        print(f"5. PyTorch (Float32) Top-1 精度: {torch_accuracy:.4f}")
        print(f"量化损失 (Float32 - INT8): {(keras_accuracy - quant_tflite_accuracy)*100:.2f}%")
    else:
        print("注意: 由于缺乏真实标签，无法计算 Top-1 精度。")

# =================================================================
# 5. 主执行逻辑
# =================================================================

if __name__ == '__main__':
    # 1. 加载测试数据
    test_data = load_test_data(NUM_TEST_SAMPLES)
    test_data_for_tf = [(np.expand_dims(np.transpose(data[0], (1,2,0)), axis=0), data[1]) for data in test_data]
    if not test_data:
        print("错误: 未加载到任何测试数据。请检查 IMAGENET_VAL_DIR 路径。")
    else:
        # 2. QTFLite Interpreter 推理 (量化模型)
        tflite_outputs = run_tflite_inference(TFLITE_MODEL_PATH, test_data_for_tf)
        quant_tflite_outputs = run_tflite_inference(Q_TFLITE_MODEL_PATH, test_data_for_tf)

        
        # load model
        torch_model = fbnet('fbnet_a', pretrained=True)
        torch_model.eval()
        torch_outputs = run_torch_inference(torch_model, test_data)

        # 3. Keras 推理 (Float32 基准)
        keras_outputs = run_keras_inference(keras_model, test_data_for_tf)

        
        # 4. 报告结果
        evaluate_and_report(tflite_outputs, keras_outputs, torch_outputs, quant_tflite_outputs, test_data)
        
    print("\n✅ 量化模型验证流程结束。")