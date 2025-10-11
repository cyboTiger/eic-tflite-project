import numpy as np
import onnxruntime as ort
import torch
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from mobile_cv.model_zoo.models.preprocess import get_preprocess



onnx_model_path = '/home/ruihan/eic/deploy/mobile_vision/my_model.onnx'

# onnx model
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# torch model
model_name = "fbnet_a"
model = fbnet(model_name, pretrained=True)
model.eval()

input_shape = (1, 3, 128, 128)

for i in range(10):
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    input_feed = {input_name: dummy_input}
    onnx_output = session.run(output_names, input_feed)
    result_onnx = onnx_output[0]

    with torch.no_grad():
        result_torch = model(torch.from_numpy(dummy_input)).cpu().numpy()

    mse = np.mean((result_onnx - result_torch) ** 2)

    print(f'MES Loss for {i}-th epoch: {mse.item()}')



