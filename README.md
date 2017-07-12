# Fast Face Swap
Inherit from [YuvalNirkin/face\_swap](https://github.com/YuvalNirkin/face_swap), this is a boosted version.

# Overview
The pipeline of the framework is modified into a more efficient one. Specifically, we have the following contributions:

- Merge expression regression into ResNet-101, which was used to regress shape and texture as illustrated in [CNN3DMM](http://www.openu.ac.il/home/hassner/projects/CNN3DMM/). Hence, we completely throw away the computation overhead of expression approximation in the original pipeline. We fine-tuned the network with images from LFW and MegaFace.

- Replace dlib face detection in two ways:

1. Use YOLO instead (please refer to the `ailabs` branch)
2. Count on KCF tracking with dlib face detection (please refer to the `KCF` branch)

- We make it a shared library and develop a python wrapper for the ease-of-use.

## Dependencies
| Library                                                            | Minimum Version | Notes                                    |
|--------------------------------------------------------------------|-----------------|------------------------------------------|
| [Boost](http://www.boost.org/)                                     | 1.47            |                                          |
| [OpenCV](http://opencv.org/)                                       | 3.0             |                                          |
| [face_segmentation](https://github.com/YuvalNirkin/face_segmentation) | 0.9          |                                          |
| [Caffe](https://github.com/BVLC/caffe)                             | 1.0             |☕️                                        |
| [Eigen](http://eigen.tuxfamily.org)                                | 3.0.0           |                                          |
| [GLEW](http://glew.sourceforge.net/)                               | 2.0.0           |                                          |
| [Qt](https://www.qt.io/)                                           | 5.4.0           |                                          |
| [HDF5](https://support.hdfgroup.org/HDF5/)                         | 1.8.18          |                                          |
| [KCFcpp](https://github.com/RudyChin/KCFcpp)                       | master          |                                          |
| [yolo.so](https://gitlab.corp.ailabs.tw/lab/py_yolo)               | so              |  The so branch.                          |

## Installation
    mkdir build
    cd build
    cmake -DCMAKE_CXX_STANDARD=14 ..
    make
    make install

- Download the [landmarks model file](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Download the [face_seg_fcn8s.zip](https://github.com/YuvalNirkin/face_segmentation/releases/download/0.9/face_seg_fcn8s.zip)
- dfm_cnn_resnet_101.zip is under `/data/rudychin/face-swap/data/models`

## Usage
Please modify the path within `py_face_swap/pytest.py` before you run it.

## Related projects
- [Face Swap](https://github.com/YuvalNirkin/face_swap), the original version
- [Deep face segmentation](https://github.com/YuvalNirkin/face_segmentation), used to segment face regions in the face swapping pipeline.
- [CNN3DMM](http://www.openu.ac.il/home/hassner/projects/CNN3DMM/), used to estimate 3D face shapes from single images.
