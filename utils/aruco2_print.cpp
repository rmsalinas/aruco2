#include "opencv2/objdetect/aruco2.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
using namespace std;
int main(int argc,char **argv) {
    try{
        if(argc!=3)throw std::runtime_error("Usage: type [m (marker)|d (diamond)|f (fractal)|b (boad)| r (raruco)] out.png");

        std::string type= argv[1];
        if(type!="m" && type!="d" && type!="f" && type!="b"&& type!="r")throw std::runtime_error("Usage: type [m (marker)|d (diamond)|f (fractal)|b (boad)] out.png");

        cv::Mat outImage;
        if(type=="m"){
            int id=0;
            cv::aruco2::getFiducialMarkerImage(outImage,cv::aruco2::DICT_ARUCO_MIP_36h12,id);
        }
        else if(type=="d"){
            cv::Vec4i id{0,1,2,3};
            cv::aruco2::getDiamondImage(outImage,cv::aruco2::DICT_ARUCO_MIP_36h12,id);
        }
        else if(type=="f"){
            cv::aruco2::getFractalMarkerImage(outImage,cv::aruco2::FRACTAL_3L_6);
        }
        else if(type=="b"){
            cv::aruco2::getGridBoardImage(outImage,cv::Size(9,5), cv::aruco2::DICT_ARUCO_MIP_36h12);
        }
        else if(type=="r"){
            int id=0;
            cv::aruco2::getRArucoMarkerImage(outImage,cv::aruco2::DICT_APRILTAG_16h5,0,2,30,2,1);

         }
        //save image
        cv::imwrite(argv[2],outImage);


    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }
}
