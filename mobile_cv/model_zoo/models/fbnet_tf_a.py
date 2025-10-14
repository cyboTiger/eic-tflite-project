from mobile_cv.model_zoo.models.tf_basic_blocks import *
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Input, Identity 
from tensorflow.keras.models import Model

class FBNetABackbone(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FBNetABackbone, self).__init__(**kwargs)
        
        # Helper to simplify block definition based on the PyTorch log
        def IRF(in_c, exp_c, out_c, k, s, dw_padding=1, shuffle=False, res=False, pwl_g=1):
            # Note: dw_groups is always exp_c
            return IRFBlock(in_c, exp_c, out_c, k, s, exp_c, pwl_g, dw_padding, shuffle, res)

        self.stages = {}
        # xif0_0: Initial Conv
        self.stages['xif0_0'] = ConvBNRelu(16, 3, strides=2, padding=1, use_bias=True) # (3, 16, k=3, s=2)
        # xif1_0: Identity (Placeholder/Pass-through)
        self.stages['xif1_0'] = tf.identity
        
        # --- Stage 2 ---
        # xif2_0: C_in=16, C_exp=48, C_out=24, k=3, s=2
        self.stages['xif2_0'] = IRF(16, 48, 24, 3, 2)
        # xif2_1: C_in=24, C_exp=24, C_out=24, k=3, s=1 (No PW, Res)
        self.stages['xif2_1'] = IRF(24, 24, 24, 3, 1, res=True)
        # xif2_2, xif2_3: Identity
        self.stages['xif2_2'] = tf.identity
        self.stages['xif2_3'] = tf.identity

        # --- Stage 3 ---
        # xif3_0: C_in=24, C_exp=144, C_out=32, k=5, s=2
        self.stages['xif3_0'] = IRF(24, 144, 32, 5, 2, dw_padding=2)
        # xif3_1: C_in=32, C_exp=96, C_out=32, k=3, s=1 (Res)
        self.stages['xif3_1'] = IRF(32, 96, 32, 3, 1, res=True)
        # xif3_2: C_in=32, C_exp=32, C_out=32, k=5, s=1 (No PW, Res)
        self.stages['xif3_2'] = IRF(32, 32, 32, 5, 1, dw_padding=2, res=True)
        # xif3_3: C_in=32, C_exp=96, C_out=32, k=3, s=1 (Res)
        self.stages['xif3_3'] = IRF(32, 96, 32, 3, 1, res=True)

        # --- Stage 4 ---
        # xif4_0: C_in=32, C_exp=192, C_out=64, k=5, s=2
        self.stages['xif4_0'] = IRF(32, 192, 64, 5, 2, dw_padding=2)
        # xif4_1: C_in=64, C_exp=192, C_out=64, k=5, s=1 (Res)
        self.stages['xif4_1'] = IRF(64, 192, 64, 5, 1, dw_padding=2, res=True)
        # xif4_2: C_in=64, C_exp=64, C_out=64, k=5, s=1 (Shuffle, No PW, Res, pwl_g=2)
        self.stages['xif4_2'] = IRF(64, 64, 64, 5, 1, dw_padding=2, shuffle=True, res=True, pwl_g=2)
        # xif4_3: C_in=64, C_exp=384, C_out=64, k=5, s=1 (Res)
        self.stages['xif4_3'] = IRF(64, 384, 64, 5, 1, dw_padding=2, res=True)
        # xif4_4: C_in=64, C_exp=384, C_out=112, k=3, s=1
        self.stages['xif4_4'] = IRF(64, 384, 112, 3, 1)
        # xif4_5: C_in=112, C_exp=112, C_out=112, k=5, s=1 (Shuffle, No PW, Res, pwl_g=2)
        self.stages['xif4_5'] = IRF(112, 112, 112, 5, 1, dw_padding=2, shuffle=True, res=True, pwl_g=2)
        # xif4_6: C_in=112, C_exp=336, C_out=112, k=5, s=1 (Res)
        self.stages['xif4_6'] = IRF(112, 336, 112, 5, 1, dw_padding=2, res=True)
        # xif4_7: C_in=112, C_exp=112, C_out=112, k=3, s=1 (Shuffle, No PW, Res, pwl_g=2)
        self.stages['xif4_7'] = IRF(112, 112, 112, 3, 1, shuffle=True, res=True, pwl_g=2)

        # --- Stage 5 ---
        # xif5_0: C_in=112, C_exp=672, C_out=184, k=5, s=2
        self.stages['xif5_0'] = IRF(112, 672, 184, 5, 2, dw_padding=2)
        # xif5_1: C_in=184, C_exp=1104, C_out=184, k=5, s=1 (Res)
        self.stages['xif5_1'] = IRF(184, 1104, 184, 5, 1, dw_padding=2, res=True)
        # xif5_2: C_in=184, C_exp=552, C_out=184, k=5, s=1 (Res)
        self.stages['xif5_2'] = IRF(184, 552, 184, 5, 1, dw_padding=2, res=True)
        # xif5_3: C_in=184, C_exp=1104, C_out=184, k=5, s=1 (Res)
        self.stages['xif5_3'] = IRF(184, 1104, 184, 5, 1, dw_padding=2, res=True)
        # xif5_4: C_in=184, C_exp=1104, C_out=352, k=5, s=1
        self.stages['xif5_4'] = IRF(184, 1104, 352, 5, 1, dw_padding=2)

        # xif6_0: Final Conv
        self.stages['xif6_0'] = ConvBNRelu(1504, 1, strides=1, use_bias=True) # (352, 1504, k=1, s=1)

    def call(self, x):
        for key in self.stages:
            x = self.stages[key](x)
        return x

# --- E. FBNet 完整模型 ---
class FBNetAKeras(tf.keras.Model):
    def __init__(self, num_classes=1000, **kwargs):
        super(FBNetAKeras, self).__init__(**kwargs)
        self.backbone = FBNetABackbone()
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.conv_head = Conv2D(num_classes, 1, strides=1, padding='valid', use_bias=True)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        x = self.backbone(x)
        x = self.avg_pool(x)
        
        # Conv Head requires 4D input, must reshape from (N, C) to (N, 1, 1, C)
        x = tf.expand_dims(tf.expand_dims(x, 1), 1)
        
        x = self.conv_head(x) # Conv2d(1504, 1000, kernel_size=(1, 1))
        x = self.flatten(x) # Final output shape (N, 1000)
        return x
    
