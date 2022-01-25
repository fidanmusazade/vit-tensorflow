import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from model.patches import PatchCreator, PatchEncoder


class MLPBlock(keras.layers.Layer):
    '''
    MLP with 2 hidden layers (since transformer units and mlp head both
    have length of 2)
    Input:
        hidden_units - number of nodes to Dense layer
        dropout_rate - rate of dropout
    Ouput:
        x - data after passing through MLP
    '''
    def __init__(self, hidden_units, dropout_rate):
        super(MLPBlock, self).__init__()
        self.linear_1 = layers.Dense(hidden_units[0])
        self.linear_2 = layers.Dense(hidden_units[1])
        self.dropout = layers.Dropout(dropout_rate)


    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = tf.nn.gelu(x)
        return self.dropout(x)


class VisionTransformer(tf.keras.Model):
    '''
    Vision Transformer (ViT) model. 
    Input:
        num_classes - number of classes (labels) in input data
        image_size - images are resized to this size, 
                     used to calculate number of patches
        patch_size - size of a patch
        projection_dim - projection dimension for Attention and Transformer
        num_heads - number of attention heads
        transformer_layers - number of transformer blocks
        data_augmentation - augmentations to apply to the input image
    Ouput:
        logits - data after passing through ViT (probabilities)
    '''
    def __init__(self, 
                num_classes, 
                image_size, 
                patch_size, 
                projection_dim, 
                num_heads, 
                transformer_layers, 
                data_augmentation,
                dropout_rate = 0.1):
        super(VisionTransformer, self).__init__()

        #inputs definition
        self.num_classes = num_classes
        self.image_size =image_size
        self.patch_size = patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = projection_dim
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]
        self.mlp_head_units = [2048, 1024] 
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.data_augmentation = data_augmentation

        #layers definition
        self.patch_creator = PatchCreator(self.patch_size)
        self.patch_encoder = PatchEncoder(self.num_patches, self.projection_dim)
        self.logits = layers.Dense(self.num_classes)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.projection_dim, dropout=dropout_rate)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp_transformer = MLPBlock(self.transformer_units, dropout_rate)
        self.mlp_head = MLPBlock(self.mlp_head_units, dropout_rate)

    
    def call(self, x):
        #augment input
        augmented = self.data_augmentation(x)
        #create and encode patches
        patches = self.patch_creator(augmented)
        encoded_patches = self.patch_encoder(patches)
        #transformer blocks
        for _ in range(self.transformer_layers):
            x1 = self.layernorm1(encoded_patches)
            attention_output = self.mha(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = self.layernorm2(x2)
            x3 = self.mlp_transformer(x3)
            encoded_patches = layers.Add()([x3, x2])
        representation = self.layernorm3(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        #final mlp part
        features = self.mlp_head(representation)
        logits = self.logits(features)
        return logits