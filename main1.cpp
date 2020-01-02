/*
 * File Type:     C/C++
 * Author:        Hutao {hutaonice@gmail.com}
 * Creation:      星期二 13/08/2019 19:41.
 * Last Revision: 星期一 19/08/2019 23:37.
 */

#include <iostream>
#include <fstream>
#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_nvJPEG.hxx"
#include <opencv2/opencv.hpp>
using namespace std;

class NVJpegDecoder{
private:
    nvjpegJpegState_t jpeg_state;
    nvjpegHandle_t handle;
    cudaStream_t stream;

    int nComponents, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t subsampling;
    nvjpegImage_t dest;
    nvjpegImage_t isz;
    unsigned char* buf;
    int capacity;
public:
    NVJpegDecoder(){
        checkCudaErrors(nvjpegCreateSimple(&handle));
        checkCudaErrors(nvjpegJpegStateCreate(handle, &jpeg_state));
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
            dest.channel[c] = NULL;
            dest.pitch[c] = 0;
            isz.pitch[c] = 0;
        }
        buf = NULL;
        capacity = 0;
    }

    void GetImageInfo(int &height, int &width, int &channel, const unsigned char* data, const int size){
        checkCudaErrors(nvjpegGetImageInfo(handle, (const uchar*)data, size, &nComponents, &subsampling, widths, heights));
        checkCudaErrors(cudaStreamSynchronize(stream));
        width = widths[0];
        height = heights[0];
        channel = nComponents;
    }

    void Decode(uchar* cvmat_data, const int height, const int width, const unsigned char* data, const int size, const bool is_bgr=true){
        int aw = 3*width;
        int ah = height;
        int sz = aw*ah;
        dest.pitch[0] = aw;
        if(sz > isz.pitch[0]){
            if(dest.channel[0]){
                checkCudaErrors(cudaFree(dest.channel[0]));
            }
            checkCudaErrors(cudaMalloc((void**)&dest.channel[0], sz));
            isz.pitch[0] = sz;
        }
        auto type = NVJPEG_OUTPUT_BGRI;
        if(!is_bgr) type = NVJPEG_OUTPUT_RGBI;
        checkCudaErrors(nvjpegDecode(handle, jpeg_state, (const uchar*)data, size, type, &dest, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaMemcpy2D(cvmat_data, width*3, dest.channel[0], dest.pitch[0], width*3, height, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    cv::Mat imread(string img_path){
        ifstream pic(img_path, ios::in|ios::binary|ios::ate);
        int size = pic.tellg();
        pic.seekg(0, ios::beg);
        if(size > capacity) {
            buf = new uchar[size];
            capacity = size;
        }
        pic.read((char*)buf, size);
        pic.close();
        int height, width, channel;
        GetImageInfo(height, width, channel, buf, size);
        cv::Mat ret(height, width, CV_8UC3);
        Decode(ret.data, height, width, buf, size, true);
        return ret;
    }

    ~NVJpegDecoder(){
        if(dest.channel[0]){
            checkCudaErrors(cudaFree(dest.channel[0]));
        }
        checkCudaErrors(nvjpegJpegStateDestroy(jpeg_state));
        checkCudaErrors(nvjpegDestroy(handle));
    }
};
int main()
{
    string img_path = "../720P.jpg";
    NVJpegDecoder decoder;
    cv::Mat img;
    int test_times = 100;
    img = decoder.imread(img_path);
    double st = cv::getTickCount();
    for(int i=0;i<test_times;i++){
        img = cv::imread(img_path);
    }
    st = cv::getTickCount() - st;
    cout<<st / cv::getTickFrequency() * 1000 / test_times<<" ms average. turbojpeg"<<endl;

    img = cv::imread(img_path);
    st = cv::getTickCount();
    for(int i=0;i<test_times;i++){
        img = decoder.imread(img_path);
    }
    st = cv::getTickCount() - st;
    cout<<st / cv::getTickFrequency() * 1000 / test_times<<" ms average. NVJPEG"<<endl;
    cv::imshow("", img);
    cv::waitKey(0);

    return 0;
}






/* EOF */

