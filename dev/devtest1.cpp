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

    cv::Mat marker;
    cv::aruco2::getRArucoMarkerImage(marker,cv::aruco2::DICT_RARUCO_16h4,1,0,3);

    cv::imwrite("image.png",marker);//exit(0);
    //open image and detect Raruco
    cv::Mat image=cv::imread(argv[1]);
    auto raruco=cv::aruco2::detectRArucoMarkers(image);
    std::cout<<"Detected "<<raruco.size()<<std::endl;
    cv::aruco2::drawFiducialMarkers(image,raruco,cv::Scalar(255,255,0));
    //resize image
    int w=1920;
    cv::resize(image,image,cv::Size(w, image.rows* w /image.cols));
    cv::imshow("image",image);
    cv::waitKey(0);
}
