import pyfaceswap
import cv2
import sys
import time

def main():
    landmarks = '/root/face_swap/data/models/shape_predictor_68_face_landmarks.dat'       # path to landmarks model file
    model_3dmm_h5 = '/root/face_swap/data/models/BaselFaceModel_mod_wForehead_noEars.h5'  # path to 3DMM file (.h5)
    model_3dmm_dat = '/root/face_swap/data/models/BaselFace.dat'                          # path to 3DMM file (.dat)
    reg_model = '/root/face_swap/data/models/3dmm_cnn_resnet_101.caffemodel'              # path to 3DMM regression CNN model file (.caffemodel)
    reg_deploy = '/root/face_swap/data/models/3dmm_cnn_resnet_101_deploy.prototxt'        # path to 3DMM regression CNN deploy file (.prototxt)
    reg_mean = '/root/face_swap/data/models/3dmm_cnn_resnet_101_mean.binaryproto'         # path to 3DMM regression CNN mean file (.binaryproto)
    seg_model = '/root/face_swap/data/models/face_seg_fcn8s.caffemodel'                   # path to face segmentation CNN model file (.caffemodel)
    seg_deploy = '/root/face_swap/data/models/face_seg_fcn8s_deploy.prototxt'             # path to face segmentation CNN deploy file (.prototxt)
    source = '/root/face_swap/data/images/brad_pitt_01.jpg'     # source image
    target = '/root/face_swap/data/images/795.jpg'  # target image

    fs = pyfaceswap.PyFaceSwap()
    if( fs.initCtx(len(sys.argv), sys.argv) ):
        print 'Initialization failed!'
        return
    fs.loadModels(landmarks, model_3dmm_h5, model_3dmm_dat, reg_model, reg_deploy,\
            reg_mean, seg_model, seg_deploy, 0, 1, 1)

    sourceImg = cv2.imread(source)
    targetImg = cv2.imread(target)

    sourceImg = cv2.resize(sourceImg, (240,240))
    targetImg = cv2.resize(targetImg, (240,240))
    
    result = []

    #cv2.imshow('Image', sourceImg)
    #cv2.waitKey(0)
    #cv2.imshow('Image', targetImg)
    #cv2.waitKey(0)

    start = time.time()
    for _ in range(5):
        if ( fs.setSourceImg(sourceImg) ):
            print 'Set Source Image Failed!'
            return
        if ( fs.setTargetImg(targetImg) ):
            print 'Set Target Image Failed!'
            return
        result = fs.swap()
    latency = (time.time() - start) / 5.0
    print latency

    cv2.imwrite('/root/face_swap/data/outptu/test.jpg', result)

    del fs

if __name__ == '__main__':
    main()

