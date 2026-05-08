#include "aruco2.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;

int testMarker1(){

    cv::Mat image;
    cv::aruco2::generateMarkerImage(image,cv::aruco2::DICT_ARUCO_MIP_36h12,1,20);

    auto markers=cv::aruco2::detectMarkers(image,cv::aruco2::DICT_ARUCO_MIP_36h12);
    if(markers.size()!=1){
        std::cerr<<"error in marker detection, detected "<<markers.size()<<" markers"<<std::endl;
        return 0;
    }
    std::cout<<"detected marker id: "<<markers[0].id<<std::endl;
    //convert image to bgr
    cv::cvtColor(image,image,cv::COLOR_GRAY2BGR);
    //draw
    cv::aruco2::drawDetectedMarkers(image,markers);


    cv::Mat  objPoints, imgPoints;
    cv::aruco2::getSolvePnpPoints(markers[0],imgPoints,objPoints);
    cout<<imgPoints<<endl;
    cout<<objPoints<<endl;


    cv::imshow("image",image);
    cv::waitKey(0);
    return 1;
}

int testMarker2(){

    cv::Mat image1,image2;
    cv::aruco2::generateMarkerImage(image1,cv::aruco2::DICT_ARUCO_MIP_36h12,1,20);
    cv::aruco2::generateMarkerImage(image2,cv::aruco2::DICT_5X5_100,2,20);
    //create an image with both
    int w=image1.cols+image2.cols;
    int h=max(image1.rows,image2.rows);
    cv::Mat image(h,w,CV_8UC1,cv::Scalar(255));
     image1.copyTo(image(cv::Rect(0,0,image1.cols,image1.rows)));
     image2.copyTo(image(cv::Rect(image1.cols,0,image2.cols,image2.rows)));


     auto markers=cv::aruco2::detectMarkers(image,{cv::aruco2::DICT_ARUCO_MIP_36h12,cv::aruco2::DICT_5X5_100});
    if(markers.size()!=2){
        std::cerr<<"error in marker detection, detected "<<markers.size()<<" markers"<<std::endl;
        return 0;
    }
    std::cout<<"detected marker id: "<<markers[0].id<<std::endl;
    std::cout<<"detected marker id: "<<markers[1].id<<std::endl;
    //convert image to bgr
    cv::cvtColor(image,image,cv::COLOR_GRAY2BGR);
    //draw
    cv::aruco2::drawDetectedMarkers(image,markers);



    cv::imshow("image",image);
    cv::waitKey(0);
    return 1;
}

int testBoard(){
    cv::Mat image;
    cv::aruco2::generateBoardImage(image, cv::Size(9,5), cv::aruco2::DICT_ARUCO_MIP_36h12 );
    cv::aruco2::Board board;
    if( cv::aruco2::detectBoard(image,cv::Size(9,5),cv::aruco2::DICT_ARUCO_MIP_36h12,board)){
        //make image bgr
        cv::cvtColor(image,image,cv::COLOR_GRAY2BGR);
        cv::aruco2::drawDetectedBoard(image,board,cv::Scalar(0,0,255),true);
        std::cout<<"Detected "<<std::endl;
        cv::Mat  objPoints, imgPoints;
        cv::aruco2::getSolvePnpPoints(board,imgPoints,objPoints);
        cout<<imgPoints<<endl;
        cout<<objPoints<<endl;
    }
    else return 0;
    cv::imshow("board",image);
    cv::waitKey(0);
}
int testDiamond(){

    cv::Mat image;
    cv::aruco2::generateDiamondImage(image,cv::aruco2::DICT_ARUCO_MIP_36h12,cv::Vec4i(0,1,2,3));
     auto diamonds=cv::aruco2::detectDiamonds(image,cv::aruco2::DICT_ARUCO_MIP_36h12);
    if(diamonds.size()==0) return 1;
     std::cout<<"detected diamond id: "<<diamonds[0].id<<std::endl;
     //draw
     cv::cvtColor(image,image,cv::COLOR_GRAY2BGR);
     cv::aruco2::drawDetectedDiamonds(image,diamonds,cv::Scalar(0,255,0),true);
     cv::Mat  objPoints, imgPoints;
     cv::aruco2::getSolvePnpPoints(diamonds[0],imgPoints,objPoints);
     cout<<imgPoints<<endl;
     cout<<objPoints<<endl;
     cv::imshow("diamond",image);
     cv::waitKey(0);
     return 1;

}

int testFractal(){
    cv::Mat image;
    cv::aruco2::generateFractalImage(image, cv::aruco2::FRACTAL_3L_6);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 1.0);
    // cv::waitKey(0);
    auto fractals = cv::aruco2::detectFractals(image, cv::aruco2::FRACTAL_3L_6);
    if(fractals.size() ==0){
        std::cerr << "error in fractal detection, detected " << fractals.size() << " markers" << std::endl;
        return 0;
    }
    std::cout << "detected fractal id: " << fractals[0].id << std::endl;

    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    cv::aruco2::drawDetectedFractals(image, fractals);

    cv::Mat objPoints, imgPoints;
    cv::aruco2::getSolvePnpPoints(fractals[0], imgPoints, objPoints);
    std::cout << imgPoints << std::endl;
    std::cout << objPoints << std::endl;

    cv::imshow("fractal", image);
    cv::waitKey(0);
    return 1;
}

int main(){
    if(testFractal()==0)
        std::cerr<<"error in test fractal"<<std::endl;

    // if(testDiamond()==0)
    //     std::cerr<<"error in test diamond"<<std::endl;
    // if(testMarker1()==0)
    //     std::cerr<<"error in test marker"<<std::endl;

    // if(testBoard()==0)
    //     std::cerr<<"error in test board"<<std::endl;

    // if(testMarker2()==0)
    //     std::cerr<<"error in test marker"<<std::endl;

}
