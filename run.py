import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from model.patches import PatchCreator, PatchEncoder
from model.vit import VisionTransformer
from model.plot import plot_random_image_patch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--num_classes", default=100, type=int)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--image_size", default=72, type=int)
    parser.add_argument("--patch_size", default=12, type=int)
    parser.add_argument("--projection_dim", default=64, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--transformer_layers", default=8, type=int)
    parser.add_argument("--plot_img", default=False, type=bool)
    args = parser.parse_args()

# Retrieve data
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


# Define data augmentation
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(args.image_size, args.image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

# Plot example patches
if args.plot_img:
	plot_random_image_patch(x_train, args.image_size, args.patch_size)

# Declare an instance of the model, optimizer, losses, etc.
vit_classifier = VisionTransformer(args.num_classes, 
                                   args.image_size, 
                                   args.patch_size, 
                                   args.projection_dim, 
                                   args.num_heads, 
                                   args.transformer_layers, 
                                   data_augmentation)
optimizer = tfa.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)

vit_classifier.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ]
)

checkpoint_filepath = "checkpoint/"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

# Train the model
history = vit_classifier.fit(
    x=x_train,
    y=y_train,
    batch_size=args.batch_size,
    epochs=args.num_epochs,
    validation_split=0.1,
    callbacks=[checkpoint_callback],
)

# Test the model
vit_classifier.load_weights(checkpoint_filepath)
_, accuracy, top_5_accuracy = vit_classifier.evaluate(x_test, y_test)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
