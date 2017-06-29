#include <string>
#include <Python.h>
#include "face_swap/face_swap.h"

// OpenGL
#include <GL/glew.h>
// Qt
#include <QApplication>
#include <QOpenGLContext>
#include <QOffscreenSurface>

//using namespace boost::python;
using std::string;

class PyFaceRenderer {

    public:
        PyFaceRenderer();
        ~PyFaceRenderer();
        int createCtx(int argc, PyObject *arglst);
        PyObject* swap(PyObject *fs);

    private:
        QApplication *a = NULL;
        QSurfaceFormat *surfaceFormat = NULL;
        QOpenGLContext *openGLContext = NULL;
        QOffscreenSurface *surface = NULL;
        face_swap::FaceRenderer *m_face_renderer = NULL;
};
