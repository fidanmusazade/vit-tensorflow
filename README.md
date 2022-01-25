# Implementation of Vision Transformers using TensorFlow

## Project Overview
The aim of the project is to implement Vision Transformers (ViT), originally described in paper “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The architecture implemented is described in the figure below:

![ViT](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)

## How to Run

The implementation of ViT will be available under the folder [model](https://github.com/GWU-Nerual-Networks/vit-tensorflow/tree/main/model). To run the model do the following:

```console
git clone git@github.com:GWU-Nerual-Networks/vit-tensorflow.git
cd vit-tensorflow
pip install -r requirements.txt
python run.py
```

If you wish, you can modify the following parameters:

| Parameter      | Meaning       | Default value       |
| -------------- | ------------- | ------------- |
| learning_rate  | Learning rate to use while training the model  | 0.001 |
| num_classes  | Number of classes in your data  | 100 |
| weight_decay  | Amount of regularization to apply  | 0.0001 |
| batch_size  | Batch size to use | 256 |
| num_epochs  | Number of epochs to train the model | 20 |
| image_size  | Size to reshape input image to | 72 |
| patch_size  | Size of patches | 12 |
| projection_dim  | Projection dimension (dimension of output of Patch Encoder) | 64 |
| num_heads  | Number of attention heads | 4 |
| transformer_layers  | Number of transformer layers | 8 |
| plot_img  | Whether to show image as a patch (demo) | False |

## Demo Notebook

You may find playground notebook in [Kaggle](https://www.kaggle.com/fidanmusazade/vision-transformers-tf/notebook). This is where you can make some small modifications and see output instantly if you do not wish to run from the terminal.
