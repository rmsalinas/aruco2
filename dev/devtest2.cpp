#include "opencv2/objdetect/aruco2.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    cv::Mat image=cv::imread(argv[1]);
    // Current OpenCV aruco — a lot of boilerplate for a simple task
    cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::aruco::DetectorParameters params;
    params.errorCorrectionRate = 0.6; // dangerously high default — causes false positives
    cv::aruco::ArucoDetector detector(dict, params);

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> rejected;
    detector.detectMarkers(image, corners, ids, rejected);

    //how many, cout
    std::cout << "Detected " << ids.size() << " markers" << std::endl;
}
