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

    //generate an april tag image
    cv::Mat outImage;
    cv::aruco2::generateMarkerImage(outImage,cv::aruco2::DICT_APRILTAG_36h10,0);

    auto markers= cv::aruco2::detectMarkers(outImage,cv::aruco2::DICT_APRILTAG_36h10);

    //make image bgr
    cv::cvtColor(outImage,outImage,cv::COLOR_GRAY2BGR);
    //draw
    for(auto m:markers)
        cv::aruco2::drawDetectedMarkers(outImage,{m});
    //show image
    cv::imshow("AprilTag", outImage);
     //wait for key press
    cv::waitKey(0);
}
