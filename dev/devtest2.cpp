#include "opencv2/objdetect/aruco2.hpp"
#include "aruco_dictionary.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing DICT_RARUCO_16h4..." << std::endl;
    
    // 1. Get predefined dictionary
    cv::aruco2::Dictionary dict = cv::aruco2::getPredefinedDictionary(cv::aruco2::DICT_RARUCO_16h4);
    std::cout << "Dictionary size (bits_id): " << dict.size() << " (expected: 20)" << std::endl;
    std::cout << "Dictionary markers: " << dict.bytesList.rows << " (expected: 5)" << std::endl;
    if (dict.bytesList.rows != 5) {
        std::cerr << "FAILED: Dictionary rows is not 5!" << std::endl;
        return 1;
    }
    
    // 2. Generate and detect each marker
    for (int id = 0; id < 5; ++id) {
        cv::Mat markerImage;
        // Generate marker image (with border)
        cv::aruco2::getFiducialMarkerImage(markerImage, cv::aruco2::DICT_RARUCO_16h4, id, 10, true);
        
        // Detect markers
        cv::aruco2::DetectionParameters params;
        params.errorCorrectionRate = 0.0; // exact match
        std::vector<cv::aruco2::FiducialMarker> detected = cv::aruco2::detectFiducialMarkers(markerImage, cv::aruco2::DICT_RARUCO_16h4, params);
        
        std::cout << "Marker ID " << id << ": detected " << detected.size() << " markers." << std::endl;
        if (detected.size() != 1) {
            std::cerr << "FAILED: Did not detect generated marker " << id << "!" << std::endl;
            return 1;
        }
        
        std::cout << "Detected ID: " << detected[0].id << " (expected: " << id << ")" << std::endl;
        if (detected[0].id != id) {
            std::cerr << "FAILED: Detected ID mismatch!" << std::endl;
            return 1;
        }
    }
    
    std::cout << "ALL TESTS PASSED!" << std::endl;
    return 0;
}