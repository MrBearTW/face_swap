import argparse
import random
import threading
import time
import sys
import cv2
import pyfaceswap
import numpy as np
from ailabs.jarvis import Processor
from PIL import Image

class FsProcessor(Processor):

    def __init__(self):
        super(FsProcessor, self).__init__()

    def on_start(self, session_id):
        print ">>> Processor [Video dummy: metadata] started: %d <<<" % (session_id, )
        self.counter = 0
        self.lastFrame = None
        self.firstFrame = True
        self.init_track = True

    def on_end(self, session_id):
        print ">>> Processor ended: %d <<<" % (session_id, )

    def pil2cv(self, img):
        cvImg = np.array(img)
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)
        return cvImg


    def on_image_frame(self, session_id, stream_id, t, image_frame, metadata):        

        global g_producerSem, g_resImg, g_tgtImg, g_producerEvent, g_consumerEvent, g_bypass, g_init_track
        print 'Receive Frame'

        start = time.time()
        if self.firstFrame:
            self.lastFrame = (image_frame, metadata)
            self.firstFrame = False

        if self.counter > skipped:
            self.counter = 0

        #Convert image from PIL to ndarray
        cvTgtImage = self.pil2cv(image_frame)
        tgtRszRatio = targetHeight / cvTgtImage.shape[0]

        g_producerSem.acquire()
        g_tgtImg = cv2.resize(cvTgtImage, None, None, fx=tgtRszRatio, fy=tgtRszRatio,\
                interpolation=cv2.INTER_LINEAR)
        if self.counter == 0:
            g_bypass = False
        else:
            g_bypass = True
        g_init_track = self.init_track
        g_consumerEvent.set()
        g_producerEvent.wait()
        g_producerEvent.clear()
        result = g_resImg
        failed = g_failed
        g_producerSem.release()

        if not failed:
            self.init_track = False
            if type(result) == type(None):
                self.init_track = True
            else:
                rszRes = cv2.resize(result, image_frame.size, interpolation=cv2.INTER_LINEAR)
                rszRes = cv2.cvtColor(rszRes, cv2.COLOR_BGR2RGB)
                self.counter = self.counter + 1
                self.lastFrame = (Image.fromarray(rszRes), metadata)
        else:
            self.counter = 0
        self.send_image_frame(self.lastFrame[0], self.lastFrame[1])
        print 'Latency: %f ms'%((time.time()-start)*1000)


class Renderer(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):

        global g_producerSem, g_resImg, g_tgtImg, g_producerEvent, g_consumerEvent, g_failed

        self.pfs = pyfaceswap.PyFaceSwap()
        if( self.pfs.createCtx(len(sys.argv), sys.argv) ):
            sys.stderr.write('Initialization failed!\n')

        self.pfs.loadModels(landmarks, model_3dmm_h5, model_3dmm_dat, reg_model, reg_deploy,\
                reg_mean, seg_model, seg_deploy, genericFace, highQual, gpuId)


        sourceImg = cv2.imread(source)
        srcRszRatio = targetHeight / sourceImg.shape[0]
        rszSrcImg = cv2.resize(sourceImg, None, None, fx=srcRszRatio, fy=srcRszRatio,\
                interpolation=cv2.INTER_LINEAR)
        if ( self.pfs.setSourceImg(rszSrcImg) ):
            sys.stderr.write('Set Source Image Failed!\n')

        while True:
            ## Wait for lock
            sys.stderr.write('Waiting for Rendering...\n')
            g_consumerEvent.wait()
            g_consumerEvent.clear()
            ## Render
            sys.stderr.write('Got something to render...\n')
            if ( not self.pfs.setTargetImg(g_tgtImg, g_bypass, g_init_track) ):
                g_failed = False
                g_resImg = self.pfs.swap()
            else:
                g_failed = True
            g_producerEvent.set()

            ## Release lock
            sys.stderr.write('Done\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--highQual', type=int, required=True)
    args = parser.parse_args()

    landmarks = '/root/face_swap/data/models/shape_predictor_68_face_landmarks.dat'       # path to landmarks model file
    model_3dmm_h5 = '/root/face_swap/data/models/BaselFaceModel_mod_wForehead_noEars.h5'  # path to 3DMM file (.h5)
    model_3dmm_dat = '/root/face_swap/data/models/BaselFace.dat'                          # path to 3DMM file (.dat)
    reg_model = '/root/face_swap/data/models/dfm_resnet_101.caffemodel'              # path to 3DMM regression CNN model file (.caffemodel)
    reg_deploy = '/root/face_swap/data/models/dfm_resnet_101_deploy.prototxt'        # path to 3DMM regression CNN deploy file (.prototxt)
    reg_mean = '/root/face_swap/data/models/dfm_resnet_101_mean.binaryproto'         # path to 3DMM regression CNN mean file (.binaryproto)
    seg_model = '/root/face_swap/data/models/face_seg_fcn8s.caffemodel'                   # path to face segmentation CNN model file (.caffemodel)
    seg_deploy = '/root/face_swap/data/models/face_seg_fcn8s_deploy.prototxt'             # path to face segmentation CNN deploy file (.prototxt)
    source = '/root/face_swap/data/images/brad_pitt_01.jpg'     # source image

    # Five global variables for synchronization
    g_init_track = True
    g_failed = False
    g_tgtImg = None
    g_resImg = None
    g_bypass = False
    g_producerSem = threading.Semaphore()
    g_producerEvent = threading.Event()
    g_consumerEvent = threading.Event()

    gpuId = args.gpu
    genericFace = 0
    highQual = args.highQual
    if highQual:
        print 'High Quality Enabled!'
        targetHeight = 360.0
        skipped = 0
    else:
        print 'Low Quality Enabled!'
        targetHeight = 180.0
        skipped = 0


    renderThread = Renderer()
    renderThread.start()
    FsProcessor.startToListen(args.port)

