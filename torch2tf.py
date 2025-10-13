import torch
import tensorflow as tf
import numpy as np

from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from mobile_cv.model_zoo.models.fbnet_tf import FBNetKeras, copy_weight_from_torch_to_tf

INPUT_SIZE = 128
NUM_CLASSES = 1000 # 假设 ImageNet 分类任务
pt_model_name = "fbnet_a"
input_tensor = torch.rand((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)

print(f"Loading PyTorch model: {pt_model_name}...")
pt_model = fbnet(pt_model_name, pretrained=True)
pt_model.eval() # 设置为评估模式
pt_state_dict = pt_model.state_dict() # 获取 PyTorch 模型的权重字典
print("PyTorch state_dict obtained successfully.")

tf_model = FBNetKeras(num_classes=NUM_CLASSES)

dummy_tf_input = tf.TensorSpec(shape=[None, INPUT_SIZE, INPUT_SIZE, 3], dtype=tf.float32)
# tf_model.compile(optimizer='adam', loss='categorical_crossentropy') # 编译可选
print("Keras model instantiated and built successfully.")

print("\nStarting weight migration from PyTorch to TensorFlow...")
# 假设 copy_weight_from_torch_to_tf 函数接受 Keras model 和 PyTorch state_dict
copy_weight_from_torch_to_tf(pt_state_dict, tf_model)
print("Weight migration completed.")

# --- 5. 验证 (可选但推荐) ---
print("\n--- Running Sanity Check ---")

# a) 准备输入数据
# PyTorch: (N, C, H, W)
pt_input_np = input_tensor.numpy()

# TensorFlow: (N, H, W, C). 必须转置通道！
# PyTorch tensor -> Numpy -> Transpose (0, 2, 3, 1) -> TensorFlow tensor
tf_input_np = np.transpose(pt_input_np, (0, 2, 3, 1)) 
tf_input_tensor = tf.constant(tf_input_np, dtype=tf.float32)

# b) 运行推断
with torch.no_grad():
    pt_output = pt_model(input_tensor)

pt_output = [pt_hidden_state.numpy() for pt_hidden_state in pt_output]

for i in range(len(pt_output)-1):
    pt_output[i] = np.transpose(pt_output[i], (0, 2, 3, 1))

tf_final_output, tf_output = tf_model(tf_input_tensor, training=False)
tf_output = [tf_hidden_state.numpy() for tf_hidden_state in tf_output]

# c) 比较输出
for i in range(len(pt_output)):
    print(f'pt shape {pt_output[i].shape}\ntf shape {tf_output[i].shape}')
    diff = np.max(np.abs(pt_output[i] - tf_output[i]))
    avg_diff = np.average(np.abs(pt_output[i] - tf_output[i]))
    print(f"Max absolute difference between PT and TF outputs: {diff:.6f}")
    print(f"Average absolute difference between PT and TF outputs: {avg_diff:.6f}")

# 通常，如果差异在 1e-5 或 1e-4 范围内，则认为权重复制成功。
if diff < 1e-4:
    print("✅ Success: Outputs are consistent. Weight copy verified.")
else:
    print("❌ Warning: Output difference is large. Check weight transposition and BN loading.")
