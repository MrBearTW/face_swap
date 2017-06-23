import argparse
import random

import pyfaceswap
from ailabs.jarvis import Processor

class FS_Processor(Processor):

    def __init__(self):
        super(MyProcessor, self).__init__()

    def on_start(self, session_id):
        print ">>> Processor [Video dummy: metadata] started: %d <<<" % (session_id, )

    def on_end(self, session_id):
        print ">>> Processor ended: %d <<<" % (session_id, )

    def on_image_frame(self, session_id, stream_id, time, image_frame, metadata):        
        # do algorithm processing
        if ( fs.setTargetImg(image_frame) ):
            print 'Set Source Image Failed!'
            return
        result = fs.swap()

        self.send_image_frame(None, metadata)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()

    landmarks = '/root/face_swap/data/models/shape_predictor_68_face_landmarks.dat'       # path to landmarks model file
    model_3dmm_h5 = '/root/face_swap/data/models/BaselFaceModel_mod_wForehead_noEars.h5'  # path to 3DMM file (.h5)
    model_3dmm_dat = '/root/face_swap/data/models/BaselFace.dat'                          # path to 3DMM file (.dat)
    reg_model = '/root/face_swap/data/models/3dmm_cnn_resnet_101.caffemodel'              # path to 3DMM regression CNN model file (.caffemodel)
    reg_deploy = '/root/face_swap/data/models/3dmm_cnn_resnet_101_deploy.prototxt'        # path to 3DMM regression CNN deploy file (.prototxt)
    reg_mean = '/root/face_swap/data/models/3dmm_cnn_resnet_101_mean.binaryproto'         # path to 3DMM regression CNN mean file (.binaryproto)
    seg_model = '/root/face_swap/data/models/face_seg_fcn8s.caffemodel'                   # path to face segmentation CNN model file (.caffemodel)
    seg_deploy = '/root/face_swap/data/models/face_seg_fcn8s_deploy.prototxt'             # path to face segmentation CNN deploy file (.prototxt)
    source = '/root/face_swap/data/images/brad_pitt_01.jpg'     # source image

    fs = pyfaceswap.PyFaceSwap()
    if( fs.initCtx(len(sys.argv), sys.argv) ):
        print 'Initialization failed!'
        return
    fs.loadModels(landmarks, model_3dmm_h5, model_3dmm_dat, reg_model, reg_deploy,\
            reg_mean, seg_model, seg_deploy, 0, 1, 1)

    sourceImg = cv2.imread(source)
    if ( fs.setSourceImg(sourceImg) ):
        print 'Set Source Image Failed!'
        return

    MyProcessor.startToListen(args.port)












