// C++ code
#include <vector>
#include <string>
#include <boost/python.hpp>
// template <class T>
boost::python::list toPythonList(std::vector<std::string> &vector) {
    typename std::vector<std::string>::iterator iter;

    boost::python::list listr;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        boost::python::list list;
        list.append(*iter);
        list.append("GOOD");
        listr.append(list);
    }
    return listr;
}

class PyFaceSwap {

    public:
        PyFaceSwap(string landmarks_path, string model_3dmm_h5_path, string model_3dmm_dat_path,
                string reg_model_path, string reg_deploy_path, string reg_mean_path,
                string seg_model_path, string seg_deploy_path, bool generic, bool with_expr, int gpu_device_id) {

            fs = new face_swap::FaceSwap(landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path,
                    reg_model_path, reg_deploy_path, reg_mean_path, generic, with_expr,
                    with_gpu, (int)gpu_device_id);

            fs->setSegmentationModel(seg_model_path, seg_deploy_path);
        }

        int setSourceImg( boost::python::object &img)


        boost::python::list myFuncGet(){return toPythonList(list);}

    private:
        face_swap::FaceSwap *fs;
};

// Wrapper code

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;


BOOST_PYTHON_MODULE(my)
{
    // class_<MyList>("MyList")
    //     .def(vector_indexing_suite<MyList>() );

    class_<MyClass>("MyClass")
        .def("myFuncGet", &MyClass::myFuncGet)
        // .def("myFuncSet", &MyClass::myFuncSet)
        ;
}
