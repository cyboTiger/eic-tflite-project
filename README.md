# EIC-Lab Alg Coding Test 3: Deploying Efficient Deep Neural Networks on Mobile Devices

## Model files
```bash
.
├── fbnet_a_imagenet_int8_224x224_model.tflite # The quantized models in tflite format
├── fbnet_a.keras # The models in tflite compatible model format:
└── fbnet_a.tflite # The unquantized models in tflite format
```

## Report
### how you convert the models in PyTorch format to tflite format

#### ONNX tool
I tried ONNX library first, it's a smooth process from torch to onnx. The problematic part lies in onnx to tf-compatible format. 
  
First off, the current [onnx2tf library](https://github.com/onnx/onnx-tensorflow) is not actively maintained from several years ago. While tensorflow has been continously upgrading, the incompatibility issues keeps emerging. e.g. basic backend op

```bash
BackendIsNotSupposedToImplementIt: ReduceMean version 18 is not implemented.
```

I have tried my best to modify onnx-tf and corresponding tensorflow lib locally, but still fails. I turned to alternative solution.

#### Manual design
To probe into the torch architecture of fbnet-a, I printed its debug info as below:

```python
FBNet(
  (backbone): FBNetBackbone(
    (stages): Sequential(
      (xif0_0): ConvBNRelu(
        (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (xif1_0): Identity()
      (xif2_0): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(16, 48, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48)
          (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      ...
      (xif6_0): ConvBNRelu(
        (conv): Conv2d(352, 1504, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(1504, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (head): ClsConvHead(
    (avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
    (conv): Conv2d(1504, 1000, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

Based on this, I manually write the [equivalent tensorflow model](mobile_cv/model_zoo/models/fbnet_tf_a.py) and its [basic block](mobile_cv/model_zoo/models/tf_basic_blocks.py).

Next, I implemented weight copy from torch to tensorflow model in the [basic block](mobile_cv/model_zoo/models/tf_basic_blocks.py) file. It essentially includes:

+ Conv-BN-Relu Block weight copy at [mobile_cv/model_zoo/models/tf_basic_blocks.py#L194]

+ Weight transpose of Conv2d from torch to tensorflow [mobile_cv/model_zoo/models/tf_basic_blocks.py#L148]

### the accuracies of models in PyTorch format, models in tflite compatible model format, unquantized models in tflite format, and quantized models in tflite format

### your ideas on how to further improve the model conversion and deployment pipeline, especially when you have a large number of models to convert.