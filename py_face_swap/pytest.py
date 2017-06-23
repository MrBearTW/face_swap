#import pyfaceswap
import cv2
import sys

def main():
    landmarks = '/data/rudychin/face-swap/data/models/shape_predictor_68_face_landmarks.dat'       # path to landmarks model file
    model_3dmm_h5 = '/data/rudychin/face-swap/data/models/BaselFaceModel_mod_wForehead_noEars.h5'  # path to 3DMM file (.h5)
    model_3dmm_dat = '/data/rudychin/face-swap/data/models/BaselFace.dat'                          # path to 3DMM file (.dat)
    reg_model = '/data/rudychin/face-swap/data/models/3dmm_cnn_resnet_101.caffemodel'              # path to 3DMM regression CNN model file (.caffemodel)
    reg_deploy = '/data/rudychin/face-swap/data/models/3dmm_cnn_resnet_101_deploy.prototxt'        # path to 3DMM regression CNN deploy file (.prototxt)
    reg_mean = '/data/rudychin/face-swap/data/models/3dmm_cnn_resnet_101_mean.binaryproto'         # path to 3DMM regression CNN mean file (.binaryproto)
    seg_model = '/data/rudychin/face-swap/data/models/face_seg_fcn8s.caffemodel'                   # path to face segmentation CNN model file (.caffemodel)
    seg_deploy = '/data/rudychin/face-swap/data/models/face_seg_fcn8s_deploy.prototxt'             # path to face segmentation CNN deploy file (.prototxt)
    source = '/data/rudychin/face-swap/data/images/brad_pitt_01.jpg'     # source image
    target = '/data/rudychin/face-swap/data/images/795.jpg'  # target image

    fs = pyfaceswap.PyFaceSwap()
    fs.initCtx(len(sys.argv), sys.argv)
    fs.loadModels(landmarks, model_3dmm_h5, model_3dmm_dat, reg_model, reg_deploy,\
            reg_mean, seg_model, seg_deploy, 0, 1, 0)

    sourceImg = cv2.imread(source)
    targetImg = cv2.imread(target)
    
    result = []

    cv2.imshow('Image', sourceImg)
    cv2.waitKey(0)
    cv2.imshow('Image', targetImg)
    cv2.waitKey(0)

    fs.setSourceImg(sourceImg)
    fs.setTargetImg(targetImg)
    fs.swap(result)

    cv2.imshow('Image', result)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

