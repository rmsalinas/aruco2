#include "opencv2/objdetect/aruco2.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {

    cv::Mat img=cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
    //apply adptive thresholding and save to file
    cv::Mat marker;
    cv::adaptiveThreshold(img,marker,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,23,7);
    //save
    cv::imwrite("marker.jpg",marker);

}
