/*
 * File Type:     C/C++
 * Author:        Hutao {hutaonice@gmail.com}
 * Creation:      星期五 03/01/2020 23:24.
 * Last Revision: 星期五 03/01/2020 23:24.
 */


#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <NVJpegDecoder.hpp>

using std::cout;
using std::endl;

namespace py = pybind11;

py::array_t<double> add_arrays_1d(py::array_t<double>& input1, py::array_t<double>& input2) {
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1) {
        throw std::runtime_error("dim != 1");
    }

    if (buf1.size != buf2.size) {
        throw std::runtime_error("size not equal");
    }

    auto result = py::array_t<double>(buf1.size);
    py::buffer_info buf3 = result.request();

    double* ptr1 = (double*)buf1.ptr;
    double* ptr2 = (double*)buf2.ptr;
    double* ptr3 = (double*)buf3.ptr;

    for (int i=0; i < buf1.shape[0]; i++) {
        ptr3[i] = ptr1[i] + ptr2[i];
    }
    return result;
}

PYBIND11_MODULE(libnvjpeg, m)
{
    m.doc() = "nvJPEG python wrapper";
    py::class_<NVJpegDecoder>(m, "NVJpegDecoder")
        .def(py::init<>())
        .def("imread", &NVJpegDecoder::imread);
}




/* EOF */

