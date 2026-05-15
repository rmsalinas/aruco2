#include "opencv2/objdetect/aruco2.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>
namespace fs = std::filesystem;
int main(){
     cv::Mat outImage;
    cv::aruco2::DictionaryType dictType = static_cast<cv::aruco2::DictionaryType>(cv::aruco2::DictionaryType::DICT_APRILTAG_36h11);
    cv::aruco2::generateMarkerImage(outImage,dictType,0);
       cv::imshow("AprilTag", outImage);
    cv::waitKey(0);

}