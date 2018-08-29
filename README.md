# CNN-PS

Satoshi Ikeahta. CNN-PS: CNN-based Photometric Stereo for General Non-Convex Surfaces, ECCV2018.


## Getting Started

This is a Keras implementation of a CNN for estimating surface normals from images captured under different illumination.

### Prerequisites

- Python3.5+
- Keras2.0+
- numpy
- OpenCV3

Tested on:
- Ubuntu 16.04, Python 3.5.2, Keras 2.0.3, Tensorflow(-gpu) 1.0.1, Theano 0.9.0, CUDA 8.0, cuDNN 5.0
  - CPU: Intel® Xeon(R) CPU E5-1650 v4 @ 3.60GHz × 12 , GPU: 3x GeForce GTX1080Ti, Memory 64GB

### Running the tests
For testing network (with DiLiGenT dataset), please download [DiLiGenT dataset](https://sites.google.com/site/photometricstereodata/) by Boxin Shi [1]

```
python test.py
```
The pretrained model for TensorFlow backend is included (weight_and_model.hdf5)

## Running the training
I will prepare for the training data soon...

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
This work was supported by JSPS KAKENHI Grant Number JP17H07324.

## References
[1] Boxin Shi, Zhipeng Mo, Zhe Wu, Dinglong Duan, Sai-Kit Yeung, and Ping Tan, "A Benchmark Dataset and Evaluation for Non-Lambertian and Uncalibrated Photometric Stereo", In IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2018.
