// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// C++ code
#include <string>
#include <boost/python.hpp>

// OpenGL
#include <GL/glew.h>

// Qt
#include <QApplication>
#include <QOpenGLContext>
#include <QOffscreenSurface>

#include <Python.h>
#include "pyFaceSwap.hpp"
#include "pyboostcvconverter.hpp"
#include "face_swap/face_swap.h"

//#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

using namespace boost::python;
using std::string;

int PyFaceSwap::initCtx(int argc, PyObject *arglst) {

    size_t cnt = PyList_GET_SIZE(arglst);
    char **argv = new char*[cnt + 1];
    for (size_t i = 0; i < cnt; i++) {
        PyObject *s = PyList_GET_ITEM(arglst, i);
        assert (PyString_Check(s));     // likewise
        size_t len = PyString_GET_SIZE(s);
        char *copy = new char[len + 1];
        memcpy(copy, PyString_AS_STRING(s), len + 1);
        argv[i] = copy;
    }
    argv[cnt] = NULL;

    // Intialize OpenGL context
    QApplication a(argc, argv);
    for (size_t i = 0; i < cnt; i++)
        delete [] argv[i];
    delete [] argv;

    QSurfaceFormat surfaceFormat;
    surfaceFormat.setMajorVersion(1);
    surfaceFormat.setMinorVersion(5);

    QOpenGLContext openGLContext;
    openGLContext.setFormat(surfaceFormat);
    openGLContext.create();
    if (!openGLContext.isValid()) return -1;

    QOffscreenSurface surface;
    surface.setFormat(surfaceFormat);
    surface.create();
    if (!surface.isValid()) return -2;

    openGLContext.makeCurrent(&surface);

    // Initialize GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) return -3;

    return 0;
}

void PyFaceSwap::loadModels(string landmarks_path, string model_3dmm_h5_path, string model_3dmm_dat_path,
        string reg_model_path, string reg_deploy_path, string reg_mean_path,
        string seg_model_path, string seg_deploy_path, bool generic, bool with_expr, int gpu_device_id) {

    const bool with_gpu = 1;
    fs = new face_swap::FaceSwap(landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path,
            reg_model_path, reg_deploy_path, reg_mean_path, generic, with_expr,
            with_gpu, (int)gpu_device_id);

    fs->setSegmentationModel(seg_model_path, seg_deploy_path);
}


int PyFaceSwap::setSourceImg(PyObject *img) {
    cv::Mat image, source_seg;
    image = pbcvt::fromNDArrayToMat(img);
    int ret = fs->setSource(image, source_seg);
    if (!ret) return -1;
    return 0;
}

int PyFaceSwap::setTargetImg(PyObject *img) {
    cv::Mat image, target_seg;
    image = pbcvt::fromNDArrayToMat(img);
    int ret = fs->setTarget(image, target_seg);
    if (!ret) return -1;
    return 0;
}

int PyFaceSwap::swap(PyObject *img) {
    cv::Mat rendered_img = fs->swap();
    if (rendered_img.empty()) return -1;
    img = pbcvt::fromMatToNDArray(rendered_img);
    return 0;
}

BOOST_PYTHON_MODULE(pyfaceswap)
{
    class_<PyFaceSwap>("PyFaceSwap")
        .def("initCtx", &PyFaceSwap::initCtx)
        .def("loadModels", &PyFaceSwap::loadModels)
        .def("setSourceImg", &PyFaceSwap::setSourceImg)
        .def("setTargetImg", &PyFaceSwap::setTargetImg)
        .def("swap", &PyFaceSwap::swap)
        ;
}
