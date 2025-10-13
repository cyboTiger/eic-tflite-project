import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras.models import Model

# helper tf layer
class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self.groups = groups

    def call(self, inputs):
        # inputs shape: (N, H, W, C)
        
        # 动态获取形状
        input_shape = tf.shape(inputs)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        
        # C 必须能被 groups 整除
        channels_per_group = C // self.groups
        
        # 1. Reshape: (N, H, W, groups, channels_per_group)
        x = tf.reshape(inputs, [N, H, W, self.groups, channels_per_group])
        
        # 2. Transpose: (N, H, W, channels_per_group, groups) -> Shuffle
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        
        # 3. Flatten back to: (N, H, W, C)
        output = tf.reshape(x, [N, H, W, C])
        return output
    
    def get_config(self):
        config = super(ChannelShuffle, self).get_config()
        config.update({"groups": self.groups})
        return config

# --- B. ConvBNRelu 块 ---
class ConvBNRelu(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', groups=1, activation='relu', use_bias=False, **kwargs):
        super(ConvBNRelu, self).__init__(**kwargs)
        self.conv = Conv2D(
            filters, 
            kernel_size, 
            strides=strides, 
            padding=padding, 
            groups=groups, 
            use_bias=use_bias
        )
        self.bn = BatchNormalization(epsilon=1e-05, momentum=0.1)
        self.relu = ReLU() if activation == 'relu' else tf.identity

    def call(self, x):
        sub_hidden_state = []
        x = self.conv(x)
        sub_hidden_state.append(x)
        x = self.bn(x)
        sub_hidden_state.append(x)
        x = self.relu(x)
        sub_hidden_state.append(x)
        return x, sub_hidden_state
    
    def build(self, input_shape):
        if not self.conv.built:
            # 如果 ConvBNRelu 已构建，则内部的 Conv2D 也应该已构建。
            # 如果没有，可能需要调用其 build 方法。
            self.conv.build(input_shape) 

# --- C. IRFBlock (Inverted Residual and Fused) ---
class IRFBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, 
                 dw_groups, pwl_groups, use_shuffle=False, has_res_conn=False, **kwargs):
        super(IRFBlock, self).__init__(**kwargs)
        
        self.has_res_conn = has_res_conn
        self.use_shuffle = use_shuffle
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # --- 1. PW Expansion (Optional) ---
        # 只有当 exp_channels > in_channels 时才需要 (即 MobileNet V2 的 'pw')
        if exp_channels != in_channels:
            self.pw = ConvBNRelu(exp_channels, 1, strides=1, activation='relu', use_bias=True)
        else:
            self.pw = None # 或者 Identity
            
        # --- 2. Channel Shuffle (Optional) ---
        if self.use_shuffle:
             # Based on PyTorch log, shuffle is always before DW, and groups is 2
             self.shuffle = ChannelShuffle(groups=2)
        else:
            self.shuffle = None

        # --- 3. DW Conv (Depthwise Conv) ---
        # Note: Keras DW Conv is slightly different, we use Conv2D(groups=C) for depthwise
        self.dw = ConvBNRelu(
            exp_channels, # 输出通道等于输入通道 (exp_channels)
            kernel_size, 
            strides=stride, 
            padding='same', 
            groups=dw_groups, # dw_groups == exp_channels
            activation='relu', 
            use_bias=True
        )

        # --- 4. PWL Projection (Projection Conv) ---
        # PWL 通常没有 ReLU
        self.pwl = ConvBNRelu(
            out_channels, 
            1, 
            strides=1, 
            groups=pwl_groups,
            activation=None, 
            use_bias=True
        )
        
        # --- 5. Residual Connection ---
        if self.has_res_conn:
            self.res_conn = Add()

    def call(self, inputs):
        x = inputs
        sub_hidden_states = []
        
        # 1. PW Expansion
        if self.pw is not None:
            x, _ = self.pw(x)
            sub_hidden_states.append(x)
            
        # 2. Channel Shuffle
        if self.shuffle is not None:
            x = self.shuffle(x)
            sub_hidden_states.append(x)

        # 3. DW Conv
        x, _ = self.dw(x)
        sub_hidden_states.append(x)
        
        # 4. PWL Projection
        x, _ = self.pwl(x)
        sub_hidden_states.append(x)
        
        # 5. Residual Connection
        if self.has_res_conn and self.stride == 1 and self.in_channels == self.out_channels:
            # Only connect if stride=1 AND input/output channels match
            x = self.res_conn([inputs, x])
            sub_hidden_states.append(x)
            
        return x, sub_hidden_states

# helper weight copy
def transpose_conv_weights(pt_weight, is_depthwise=False):
    """
    将 PyTorch 卷积核张量转置为 TensorFlow 格式。
    
    PyTorch 形状: [C_out, C_in, K_H, K_W] 或 [C_out, 1, K_H, K_W] (深度卷积)
    TensorFlow 形状: [K_H, K_W, C_in, C_out] 或 [K_H, K_W, C_in, C_multiplier] (深度卷积)
    """
    
    if is_depthwise:
        # 深度卷积：[C_out, 1, K_H, K_W] -> [K_H, K_W, C_out, 1] 
        # C_out 实际是 C_in
        return pt_weight.permute(2, 3, 1, 0).numpy()
    else:
        # 标准/逐点卷积：[C_out, C_in, K_H, K_W] -> [K_H, K_W, C_in, C_out]
        return pt_weight.permute(2, 3, 1, 0).numpy()

def load_bn_weights(bn_layer, pt_bn_prefix, pt_state_dict):
    """加载 BatchNorm 层的权重 (gamma, beta, mean, var)。"""
    
    # Keras BN 权重顺序: [gamma, beta, moving_mean, moving_variance]
    # PyTorch BN 权重名称: [weight, bias, running_mean, running_var]
    
    pt_weights = [
        pt_state_dict[f'{pt_bn_prefix}.weight'],
        pt_state_dict[f'{pt_bn_prefix}.bias'],
        pt_state_dict[f'{pt_bn_prefix}.running_mean'],
        pt_state_dict[f'{pt_bn_prefix}.running_var'],
    ]
    
    # 1. 检查并强制构建 BN 层 (核心修复)
    if not bn_layer.built:
        # 所有 BN 权重（gamma, beta, mean, var）的大小都等于通道数 C。
        # 我们可以从 PyTorch 权重的形状中获取 C。
        channels = pt_weights[0].shape[0] 
        
        # BN 层的输入形状只需要知道通道数 (最后一位)
        # 强制构建 BN 层
        bn_layer.build(input_shape=(None, None, None, channels)) 
        
        # 额外的检查：如果构建后权重数量仍然是 0，说明构建失败
        if len(bn_layer.get_weights()) == 0:
            raise RuntimeError(f"Failed to build BatchNormalization layer: {bn_layer.name}. Expected 4 weights, got 0 after build.")

    # 2. 设置权重
    bn_layer.set_weights([w.numpy() for w in pt_weights])

def load_conv_bn_relu_block(tf_block, pt_prefix, pt_state_dict):
    """加载 ConvBNRelu 模块的权重。"""
    
    # 强制构建 ConvBNRelu 内部的子层，以确保权重张量被创建
    # 这里的 shape 应该根据模型推理得到，但最安全的方法是基于 pt_state_dict 推断
    # 假设我们知道输入图像的尺寸是 128x128
    
    pt_conv_weight = pt_state_dict[f'{pt_prefix}.conv.weight']
    
   # 1. 确定输入通道数 C_in (用于构建 Keras 层)
    is_depthwise = tf_block.conv.groups > 1
    
    if is_depthwise and tf_block.conv.filters == tf_block.conv.groups:
        # 【关键修正】对于深度卷积 (DW Conv)，输入通道数 C_in 必须等于 groups。
        # 这里 C_in = 48。
        in_channels = tf_block.conv.groups
    elif is_depthwise:
        C_in_per_group = pt_conv_weight.shape[1] 
        in_channels = C_in_per_group * tf_block.conv.groups
    else:
        # 对于标准或逐点卷积 (PW/PWL)，输入通道数 C_in 来自 PyTorch 权重的第二个维度。
        in_channels = pt_conv_weight.shape[1] 
    
    # Keras 期待的输入形状 (N, H, W, C_in)
    dummy_input_shape = (None, None, None, in_channels)
    
    # 检查并构建 ConvBNRelu 块 (如果尚未构建)
    if not tf_block.built:
        tf_block.build(dummy_input_shape)
    
    # --- 1. Conv 权重加载 ---
    
    # 检查 Conv2D 层是否已构建
    if not tf_block.conv.built:
        # 如果 ConvBNRelu 已构建，则内部的 Conv2D 也应该已构建。
        # 如果没有，可能需要调用其 build 方法。
        tf_block.conv.build(dummy_input_shape) 

    # 确定是否为深度卷积
    is_depthwise = tf_block.conv.groups > 1

    # 权重转置
    tf_kernel = transpose_conv_weights(pt_conv_weight, is_depthwise=is_depthwise)
    
    # 确定权重列表长度 (取决于 use_bias)
    weights_list = [tf_kernel]
    if tf_block.conv.use_bias:
        # 如果 Keras 层期望偏置，则加载它
        pt_conv_bias = pt_state_dict.get(f'{pt_prefix}.conv.bias')
        if pt_conv_bias is not None:
             weights_list.append(pt_conv_bias.numpy())
        else:
             # 这通常意味着 PyTorch 模型定义和 Keras 定义不匹配
             raise ValueError(f"Keras layer {tf_block.conv.name} expects bias, but PyTorch state_dict lacks {pt_prefix}.conv.bias")
    
    # 设置权重
    tf_block.conv.set_weights(weights_list) 
    
    # --- 2. BN 权重加载 ---
    load_bn_weights(tf_block.bn, f'{pt_prefix}.bn', pt_state_dict)
