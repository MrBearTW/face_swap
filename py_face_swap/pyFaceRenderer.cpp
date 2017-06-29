// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// C++ code
#include <string>
#include <boost/python.hpp>

#include <iostream>

#include <boost/python.hpp>
#include "face_swap/face_swap.h"
#include "pyFaceRenderer.hpp"

#include "pyboostcvconverter.hpp"

using namespace boost::python;
using std::cout;
using std::string;

PyFaceRenderer::PyFaceRenderer() {
    import_array();
    m_face_renderer = new face_swap::FaceRenderer();
}

PyFaceRenderer::~PyFaceRenderer() {
    if (surface) delete surface;
    if (openGLContext) delete openGLContext;
    if (surfaceFormat) delete surfaceFormat;
    if (a) delete a;
    if (m_face_renderer) delete m_face_renderer;
}

int PyFaceRenderer::createCtx(int argc, PyObject *arglst) {

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
    a = new QApplication(argc, argv);
    for (size_t i = 0; i < cnt; i++)
        delete [] argv[i];
    delete [] argv;

    surfaceFormat = new QSurfaceFormat;
    surfaceFormat->setMajorVersion(1);
    surfaceFormat->setMinorVersion(5);

    openGLContext = new QOpenGLContext;
    openGLContext->setFormat(*surfaceFormat);
    openGLContext->create();
    if (!openGLContext->isValid()) return -1;

    surface = new QOffscreenSurface;
    surface->setFormat(*surfaceFormat);
    surface->create();
    if (!surface->isValid()) return -2;

    openGLContext->makeCurrent(surface);

    // Initialize GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) return -3;

    return 0;
}

PyObject* PyFaceRenderer::swap(PyObject *obj) {
    face_swap::FaceSwap *fs = (face_swap::FaceSwap*) obj;
    cv::Mat tgt_cropped_img = fs->getTgtCroppedImg();
    cv::Mat vecR = fs->getVecR();
    cv::Mat vecT = fs->getVecT();
    float K4 = fs->getK4();
    face_swap::Mesh dst_mesh = fs->getDstMesh();

    // Initialize renderer
    m_face_renderer->init(tgt_cropped_img.cols, tgt_cropped_img.rows);
    m_face_renderer->setProjection(K4);
    m_face_renderer->setMesh(dst_mesh);

    // Render
    cv::Mat rendered_img;
    m_face_renderer->render(vecR, vecT);
    m_face_renderer->getFrameBuffer(rendered_img);

    // Return images to be blended
    PyObject *ret = pbcvt::fromMatToNDArray(rendered_img);
    return ret;
}

