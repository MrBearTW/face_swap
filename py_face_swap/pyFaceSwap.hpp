#include <string>
#include <Python.h>
#include "face_swap/face_swap.h"

// Qt
#include <QApplication>
#include <QOpenGLContext>
#include <QOffscreenSurface>

// OpenGL
#include <GL/glew.h>

//using namespace boost::python;
using std::string;

class PyFaceSwap {

    public:
        PyFaceSwap();
        ~PyFaceSwap();
        int initCtx(int argc, PyObject *arglst);
        int setSourceImg(PyObject *pyImg);
        int setTargetImg(PyObject *pyImg);
        PyObject* swap();

        void loadModels(string landmarks_path, string model_3dmm_h5_path, string model_3dmm_dat_path,
                string reg_model_path, string reg_deploy_path, string reg_mean_path,
                string seg_model_path, string seg_deploy_path, bool generic, bool with_expr, int gpu_device_id);

    private:
        face_swap::FaceSwap *fs = NULL;
        QApplication *a = NULL;
        QSurfaceFormat *surfaceFormat = NULL;
        QOpenGLContext *openGLContext = NULL;
        QOffscreenSurface *surface = NULL;
};
