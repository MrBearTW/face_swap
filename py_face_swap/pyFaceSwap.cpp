// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// C++ code
#include <string>
#include <boost/python.hpp>

#include <iostream>

#include <boost/python.hpp>
#include "pyFaceSwap.hpp"
#include "face_swap/face_swap.h"

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include "pyboostcvconverter.hpp"
#include "pyFaceRenderer.hpp"

using namespace boost::python;
using std::cout;
using std::string;

PyFaceSwap::PyFaceSwap() {
    import_array();
}

PyFaceSwap::~PyFaceSwap() {
    if (fs) delete fs;
}

void PyFaceSwap::loadModels(string landmarks_path, string model_3dmm_h5_path, string model_3dmm_dat_path,
        string reg_model_path, string reg_deploy_path, string reg_mean_path,
        string seg_model_path, string seg_deploy_path, bool generic, bool highQual,
        int gpu_device_id) {

    const bool with_gpu = 1;
    fs = new face_swap::FaceSwap(landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path,
            reg_model_path, reg_deploy_path, reg_mean_path, generic, highQual,
            with_gpu, (int)gpu_device_id);

    fs->setSegmentationModel(seg_model_path, seg_deploy_path);
}


int PyFaceSwap::setSourceImg(PyObject* pyImg) {
    cv::Mat image, source_seg;
    image = pbcvt::fromNDArrayToMat(pyImg);
    //cv::imshow("Source", image);
    //cv::waitKey(0);
    int ret = fs->setSource(image, source_seg);
    if (!ret) return -1;
    return 0;
}

int PyFaceSwap::setTargetImg(PyObject* pyImg, bool bypass) {
    cv::Mat image, target_seg;
    image = pbcvt::fromNDArrayToMat(pyImg);
    //cv::imshow("Target", image);
    //cv::waitKey(0);
    int ret = fs->setTarget(image, target_seg, bypass);
    if (!ret) return -1;
    return 0;
}

PyObject* PyFaceSwap::getFs() {
    return (PyObject*)fs;
}

PyObject* PyFaceSwap::blend(PyObject *pyImg) {
    cv::Mat rendered_img, blended_img;
    rendered_img = pbcvt::fromNDArrayToMat(pyImg);

    cv::Mat target_img = fs->getTargetImg();
    cv::Mat target_seg = fs->getTargetSeg();
    cv::Rect target_bbox = fs->getTargetBbox();

    cv::Mat tgt_rendered_img = cv::Mat::zeros(target_img.size(), CV_8UC3);
    rendered_img.copyTo(tgt_rendered_img(target_bbox));

    blended_img = fs->blend(tgt_rendered_img, target_img, target_seg);
    PyObject *ret = pbcvt::fromMatToNDArray(blended_img);
    return ret;
}

static void init_ar(){
    Py_Initialize();
    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}


BOOST_PYTHON_MODULE(pyfaceswap)
{
    init_ar();
    class_<PyFaceSwap>("PyFaceSwap", init<>())
        .def(init<>())
        .def("loadModels", &PyFaceSwap::loadModels)
        .def("setSourceImg", &PyFaceSwap::setSourceImg)
        .def("setTargetImg", &PyFaceSwap::setTargetImg)
        .def("blend", &PyFaceSwap::blend)
        .def("getFs", &PyFaceSwap::getFs)
        ;
    class_<PyFaceRenderer>("PyFaceRenderer", init<>())
        .def(init<>())
        .def("createCtx", &PyFaceRenderer::createCtx)
        .def("swap", &PyFaceRenderer::swap)
        ;
}
