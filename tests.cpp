#include "opencv2/objdetect/aruco2.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/flann.hpp>
using namespace std;

// int testFLANN(){
//     std::vector<cv::Point2f> corners={{3679.83,1857.22},{3324.43,1850.67},{3278.83,1502.32},{3621.32,1508.69},{3662.26,1837.97},{3336.06,1839.28},{3291.74,1516.03},{3608.94,1518.72},{2980.66,1843.22},{2628.11,1837.14},{2607.73,1491.57},{2948.22,1496.76},{2289.31,1830.19},{1943.99,1824.17},{1945.51,1482.34},{2280.68,1487.39},{1607.47,1819.11},{1260.04,1813.26},{1283.94,1470.34},{1620.03,1475.85},{923.063,1808.08},{576.193,1802.4},{624.713,1459.78},{959.715,1464.93},{3270.25,1494.98},{2952.95,1492.19},{2922.75,1190.02},{3231.05,1194.81},{3284.82,1507.62},{2942.74,1502.59},{2912.32,1178.33},{3242.98,1183.19},{2612.82,1496.37},{2276.44,1491.63},{2267.61,1170.13},{2593.64,1174.43},{1949.73,1486.61},{1614.92,1480.64},{1626.77,1160.34},{1952,1165.51},{1289.38,1476.21},{953.709,1470.21},{986.779,1148.92},{1311.98,1154.89},{3567.28,1195.1},{3237.51,1189.03},{3197.63,884.15},{3518.61,888.76},{2918.31,1183.62},{2588.69,1179.37},{2570.64,875.672},{2889.87,878.902},{2271.97,1174.25},{1947.62,1169.61},{1948.56,868.197},{2263.8,872.533},{1630.99,1164.6},{1306.74,1159.52},{1327.81,858.358},{1642.69,862.765},{992.896,1155.52},{666.107,1149.96},{705.246,845.776},{1023.25,851.582},{3204.48,889.981},{2883.49,885.252},{2859.31,593.752},{3175.43,594.736},{2575.86,880.331},{2259.44,876.647},{2252.33,589.235},{2560.47,592.332},{1954.36,873.708},{1636.7,868.068},{1646.93,580.76},{1956.14,584.797},{1332.02,862.624},{1017.11,856.71},{1045.32,572.178},{1350.95,577.635},{3481.36,605.324},{3169.05,601.083},{3138.84,324.116},{3446.12,325.545},{2866.14,599.61},{2555.38,597.142},{2541.51,320.931},{2845.5,322.419},{2256.79,593.253},{1950.17,590.12},{1951.8,317.445},{2250.88,319.418},{1654.16,587.663},{1344.83,582.787},{1362.8,309.253},{1663.89,313.056},{1050.94,577.867},{741.476,571.648},{776.433,300.444},{1076.97,304.56},{3327.05,1847.65},{2977.74,1840.48},{2945.48,1499.68},{3281.82,1504.97},{2630.85,1834.22},{2287.2,1828.06},{2278.56,1489.51},{2610.64,1494.31},{1946.11,1822.05},{1604.75,1816.18},{1617.11,1478.59},{1947.62,1484.48},{1262.96,1810.53},{920.46,1805.05},{956.712,1467.57},{1286.66,1473.27},{3618.7,1511.71},{3281.82,1504.97},{3240.24,1186.11},{3564.22,1192.53},{2945.48,1499.68},{2610.64,1494.31},{2591.52,1176.55},{2915.32,1180.97},{2278.56,1489.51},{1947.62,1484.48},{1949.81,1167.56},{2269.79,1172.18},{1617.11,1478.59},{1286.66,1473.27},{1308.98,1157.53},{1628.88,1162.47},{956.712,1467.57},{627.315,1462.81},{669.943,1146.75},{989.494,1151.86},{3240.25,1186.11},{2915.32,1180.98},{2887.03,881.727},{3200.69,886.725},{2591.52,1176.55},{2269.79,1172.19},{2261.62,874.587},{2573.63,878.324},{1949.81,1167.55},{1628.88,1162.47},{1640.44,864.751},{1950.74,870.259},{1308.98,1157.53},{989.494,1151.86},{1019.41,854.787},{1329.92,860.49},{3515.88,891.68},{3200.69,886.725},{3171.89,598.263},{3478.26,602.785},{2887.04,881.725},{2573.63,878.324},{2558.28,594.392},{2863.1,597.005},{2261.62,874.587},{1950.74,870.259},{1952.4,588.114},{2254.56,591.241},{1640.44,864.751},{1329.92,860.49},{1348.65,579.559},{1650.55,584.209},{1019.41,854.787},{708.649,849.44},{745.388,568.534},{1047.42,574.313},{3171.89,598.263},{2863.1,597.005},{2842.6,325.169},{3141.93,326.654},{2558.28,594.392},{2254.56,591.241},{2248.65,321.425},{2544.55,323.537},{1952.4,588.113},{1650.55,584.209},{1660.07,316.284},{1954.02,319.457},{1348.65,579.559},{1047.43,574.312},{1073.06,307.674},{1366.42,312.707}};

//     //now, create a flann index with these corners
//     cv::flann::KDTreeIndexParams indexParams(1);
//     cv::Mat data = cv::Mat(corners).reshape(1, static_cast<int>(corners.size()));
//     auto index=std::make_shared<cv::flann::Index>(data, indexParams);
//     //now, search for every point 4 nearest
//     int maxResults=4;
//     int nTimesError1=0;
//     int nTimesError2=0;
//     for(size_t i=0;i<corners.size();i++){
//         std::vector<int> indices(4);
//         std::vector<float> dists(4);
//         int nn=index->radiusSearch(data.row(static_cast<int>(i)), indices, dists, 100,maxResults);
//         if( nn!=indices.size() ) {nTimesError1++; }
//         if( nn>maxResults) { nTimesError2++;  }
//     }
//     std::cout<<"nTimesError1: "<<nTimesError1<<std::endl;
//     std::cout<<"nTimesError2: "<<nTimesError2<<std::endl;

//     return 1;
// }
int testMarker1(){

    cv::Mat image;
    cv::aruco2::generateMarkerImage(image,cv::aruco2::DICT_ARUCO_MIP_36h12,1,20);

    //save to file
    cv::imwrite("marker.jpg",image);
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
    //Save to file
    cv::imwrite("board.jpg",image);
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
    return 1;
}
int testDiamond(){

    cv::Mat image;
    cv::aruco2::generateDiamondImage(image,cv::aruco2::DICT_ARUCO_MIP_36h12,cv::Vec4i(0,1,2,3));
    //save to file
    cv::imwrite("diamond.jpg",image);
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
    //save to file
    cv::imwrite("fractal.jpg",image);
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

    if(testDiamond()==0)
        std::cerr<<"error in test diamond"<<std::endl;
    if(testMarker1()==0)
        std::cerr<<"error in test marker"<<std::endl;

    if(testBoard()==0)
        std::cerr<<"error in test board"<<std::endl;

    if(testMarker2()==0)
        std::cerr<<"error in test marker"<<std::endl;

}
