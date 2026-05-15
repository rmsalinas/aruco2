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
    for(int dict=0;dict<=cv::aruco2::DictionaryType::DICT_APRILTAG_36h11;dict++){
        std::cout<<"dict="<<dict<<std::endl;
        cv::Mat outImage;
        cv::aruco2::DictionaryType dictType = static_cast<cv::aruco2::DictionaryType>(dict);
        cv::aruco2::generateMarkerImage(outImage,dictType,0);
        // //go invert color in rect 40,40,20,20
        cv::imshow("AprilTag-before", outImage);
        cv::Mat roi=outImage(cv::Rect(40,40,20,20));
        cv::bitwise_not(roi,roi);
        cv::imshow("AprilTag-after", outImage);


        //rotate image 90 degs
        cv::rotate(outImage,outImage,cv::ROTATE_90_CLOCKWISE);

        auto markers= cv::aruco2::detectMarkers(outImage,dictType,{.errorCorrectionRate=0.6});

        //make image bgr
        cv::cvtColor(outImage,outImage,cv::COLOR_GRAY2BGR);
        //draw
        for(auto m:markers)
            cv::aruco2::drawDetectedMarkers(outImage,{m});
        //show image
        cv::imshow("AprilTag", outImage);
        //wait for key press
        std::cout<<"dict="<<dict<<" done"<<std::endl;
        cv::waitKey(0);
    }
}
