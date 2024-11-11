# resnet-50-img-classify

# Image Classification and Explanation using ResNet50

This project uses TensorFlow and Keras to perform image classification with the ResNet50 architecture. It applies Gradient-CAM (GradCAM) to generate visual explanations for the model's predictions.

## Features
- Image classification using ResNet50 pre-trained on ImageNet.
- Generation of GradCAM heatmaps to explain predictions.
- Support for multiple images from local paths or online URLs.
- Prediction of whether the image is "AI-generated" or "Real."

## Requirements
- TensorFlow 2.x
- Numpy
- OpenCV
- Matplotlib
- Pillow
- Requests

You can install the required libraries by running:

```bash
pip install tensorflow numpy opencv-python matplotlib pillow requests
