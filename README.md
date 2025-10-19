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

I have tried my best to modify onnx-tf and corresponding tensorflow lib locally, but still fails. So I turned to alternative solution.

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

+ Conv-BN-Relu Block weight copy at [here](mobile_cv/model_zoo/models/tf_basic_blocks.py#L194)

+ Weight transpose of Conv2d from torch to tensorflow [here](mobile_cv/model_zoo/models/tf_basic_blocks.py#L148)

During the copy process, the most tricky parts lie in: 

+ the padding of tensorflow Conv2d layer. 
  [TF Conv2d](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2D) padding argument only supports 2 options: `'valid'`, meaning no padding,  or `'same'`, meaning padding flexibly to make output size proportional to input size. Weird things will occur for some input height/width, kernel size and stride. 
  
  For example, `input_size=(224, 224), kernel_size=(3, 3), stride=(2, 2), padding='same'` in tensorflow will result in asymmetric padding between left and right side of a single image channel. While the corresponding implementation in torch `input_size=(224, 224), kernel_size=(3, 3), stride=(2, 2), padding=(1,1)` will pad evenly to two sides.

  To tackle this, for all the initialization of Conv2d in tensorflow, I adopt `padding='valid'`. When padding is required, I apply `tf.padding` manually before the Conv2d layer.

+ Weight transpose of Conv2d before real copy
  torch conv2d kernel weight shape is `[C_out, C_in, K_H, K_W]`, while tensorflow shape is `[K_H, K_W, C_in, C_out]`. So a permutation of `pt_weight.permute(2, 3, 1, 0)` is required

#### Conversion to tflite format and Verification
converting to tflite model is simple with following snippet:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

#### Verification
For verification, I prepared random input of size `(1, 3, 224, 224)` and find the difference between torch/tensorflow/TFLite model ignorable, as in [here](torch2tf_a.py)

### the accuracies of models
I validated using ImageNet2012-1k validation dataset. But it seems that even the torch model fails to reach high accuracy. I doubt it's because the label alignment issue when training the model. So I measure accuracy using PyTorch model as baseline.

| Model format | Top-1 accuracy | Top-5 accuracy |
| :--- | :--- | :--- |
| PyTorch | 100% | 100% |
| TFLite Keras | 100% | 100% | 
| TFLite FP32 | 100% | 100% | 
| TFLite INT8 | 84.62% | 98.06% | 

### ideas on further improvement ofthe model conversion and deployment pipeline
Especially when you have a large number of models to convert.

Answer: 
If these models are all CNN models, I think we could design a intermediate representation that express the hierarchy of these CNN models. It can be in the form of a json/yaml file, each item can either be a struct or a basic module (like linear, Conv, softmax, Norm, ReLU...), with hyperparameters like kernel_size clarified in the readable text file.

In this case, the challenge comes down to 
+ parse the readable file
+ design API between the text form and certain deep learning framework, e.g. PyTorch, Tensorflow, Jax...

If there are also LLM-style models, we can add transformer as a basic module inside the library. And I think it's important to make it easy to extend basic modules to customized operators, so that every developer can upload/merge their own operators to accelerate archtecture designing.