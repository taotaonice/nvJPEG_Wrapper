/*
 * File Type:     C/C++
 * Author:        Hutao {hutaonice@gmail.com}
 * Creation:      星期二 13/08/2019 19:41.
 * Last Revision: 星期六 04/01/2020 01:24.
 */

#include "NVJpegDecoder.hpp"
using namespace std;

int main()
{
    string img_path = "../imgs/720P.jpg";
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

