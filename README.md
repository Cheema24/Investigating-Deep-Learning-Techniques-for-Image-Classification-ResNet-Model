# ResNet-Style Image Classification (COCO Subset)

## Overview

This project, developed by **Uzbek Imtiaz Cheema**, implements a deep learning model for image classification using a subset of the COCO dataset. The task was simplified from object detection to classification by extracting bounding boxes and converting them into standalone image samples.

The model classifies five object categories:

* Person
* Chair
* Car
* Dining Table
* Bottle

## Objectives

* Explore CNNs for image classification
* Implement a lightweight ResNet-style architecture
* Handle class imbalance
* Evaluate performance under CPU-only constraints

## Dataset

A subset of the COCO 2017 dataset was used, producing **7397 cropped samples** with natural class imbalance.

## Model Architecture

* Convolution + batch normalisation
* Residual (skip) connections
* 3 stages (32, 64, 128 channels)
* Global average pooling
* Fully connected classifier (5 classes)

## Techniques Used

* Data augmentation (flip, colour jitter)
* Image resizing (128×128)
* Weighted sampling (imbalanced data)
* Cross-entropy loss
* Adam optimiser

## Results

* Best validation accuracy: **46.8%**
* Final evaluation accuracy: **58.7%**

## Key Learnings

* Implementing **residual (skip) connections** improves training stability in deeper networks
* Real-world datasets like COCO introduce **class imbalance challenges** that significantly affect model performance
* Techniques such as **weighted sampling and data augmentation** help but do not fully solve imbalance issues
* Model performance is heavily influenced by **class distribution and data quality**
* Training on **CPU-only environments** requires careful trade-offs in model complexity and training time
* Converting object detection data into classification tasks is a practical way to simplify complex problems

## Limitations

* Class imbalance impacts performance
* CPU-only training limits model depth
* Some classes visually similar

## Future Improvements

* Use pretrained/deeper models
* Increase image resolution
* Apply advanced imbalance methods (e.g. focal loss)

## Tech Stack

* Python
* PyTorch

## Author

**Uzbek Imtiaz Cheema**
University of Greenwich – Computer Science

---

# References

## Academic Sources

* He et al. (2016). *Deep Residual Learning for Image Recognition*
  https://arxiv.org/abs/1512.03385

* Lin et al. (2014). *Microsoft COCO: Common Objects in Context*
  https://arxiv.org/abs/1405.0312

* Buda et al. (2018). *A systematic study of the class imbalance problem in CNNs*
  https://arxiv.org/abs/1710.05381

* Goodfellow, Bengio & Courville (2016). *Deep Learning*. MIT Press
  https://www.deeplearningbook.org/

## Frameworks & Dataset

* PyTorch Documentation
  https://pytorch.org/docs/stable/

* COCO API
  https://github.com/cocodataset/cocoapi

## Implementation Resources

* Introduction to PyTorch
  https://www.geeksforgeeks.org/introduction-to-pytorch/

* CNN in PyTorch
  https://www.geeksforgeeks.org/convolutional-neural-network-in-pytorch/

* Residual Networks (ResNet) Explained
  https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/

* Handling Imbalanced Data
  https://www.geeksforgeeks.org/handling-imbalanced-data-in-machine-learning/

* WeightedRandomSampler in PyTorch
  https://www.geeksforgeeks.org/how-to-handle-imbalanced-classes-in-pytorch/

* PyTorch Image Transforms
  https://www.geeksforgeeks.org/image-transforms-in-pytorch/
