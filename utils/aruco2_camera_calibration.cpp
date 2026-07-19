/**
 * @file aruco2_camera_calibration.cpp
 * @brief Camera calibration utility using an ArUco2 GridBoard.
 *
 * This program detects an ArUco2 board in a set of calibration images
 * and computes the camera intrinsic parameters (camera matrix and distortion coefficients).
 * The resulting calibration data is saved to a YAML or XML file, which can then be
 * used for pose estimation by other utilities.
 *
 * Usage:
 *   ./aruco2_camera_calibration <image_folder> [options]
 *
 * Positional Arguments:
 *   @path         Folder containing the calibration images (default: ".")
 *
 * Options:
 *   -w=<int>      Number of marker columns in the board (default: 9)
 *   -h=<int>      Number of marker rows in the board (default: 5)
 *   -dict=<int>   Dictionary ID (default: 21 = DICT_ARUCO_MIP_36h12)
 *   -ms=<float>   Physical marker side length in meters (default: 0.05)
 *   -o=<string>   Output calibration file path (default: "calibration.yaml")
 *   -show         If true, shows each detection result in a window (default: false)
 *   --help        Show the help message
 */
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
    cv::CommandLineParser parser(argc, argv,
        "{@path     | .               | folder with calibration images (jpg/png) }"
        "{w         | 9               | board width — number of marker columns }"
        "{h         | 5               | board height — number of marker rows }"
        "{dict      | 21              | dictionary id (21 = DICT_ARUCO_MIP_36h12) }"
        "{ms        | 0.05            | physical marker side length in metres }"
        "{o         | calibration.yaml| output calibration file (YAML or XML) }"
        "{show      | false           | show each detection result in a window }"
        "{help      |                 | show this help message }");

    if (parser.has("help")) { parser.printMessage(); return 0; }

    std::string folder  = parser.get<std::string>("@path");
    int  bw             = parser.get<int>("w");
    int  bh             = parser.get<int>("h");
    int  dictId         = parser.get<int>("dict");
    float markerSize    = parser.get<float>("ms");
    std::string outFile = parser.get<std::string>("o");
    bool showImages     = parser.get<bool>("show");

    auto dict = static_cast<cv::aruco2::DictionaryType>(dictId);
    cv::Size boardSize(bw, bh);

    // --- collect image paths ---
    std::vector<std::string> paths;
    for (auto& entry : fs::directory_iterator(folder)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());

    if (paths.empty()) {
        std::cerr << "No jpg/png images found in: " << folder << "\n";
        return 1;
    }
    std::cout << "Found " << paths.size() << " images\n";

    std::vector<std::vector<cv::Point3f>> allObjPts;
    std::vector<std::vector<cv::Point2f>> allImgPts;
    cv::Size imageSize;

    for (auto& path : paths) {
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        std::cout<<"image "<<path<<std::endl;

        if (image.empty()) { std::cerr << "  [warn] cannot read: " << path << "\n"; continue; }
        if (imageSize.width == 0) imageSize = image.size();

        cv::aruco2::GridBoard board;
        if (!cv::aruco2::detectGridBoard(image, boardSize, dict, board)) {
            std::cout << "  [skip] board not found: " << path << "\n";
            continue;
        }

        cv::Mat imgPtsMat, objPtsMat;
        cv::aruco2::getSolvePnpPoints(board, objPtsMat, imgPtsMat, markerSize);

        if (imgPtsMat.empty()) { std::cout << "  [skip] no corner points: " << path << "\n"; continue; }

        std::vector<cv::Point2f> imgPts(imgPtsMat.begin<cv::Point2f>(), imgPtsMat.end<cv::Point2f>());
        std::vector<cv::Point3f> objPts(objPtsMat.begin<cv::Point3f>(), objPtsMat.end<cv::Point3f>());

        allImgPts.push_back(imgPts);
        allObjPts.push_back(objPts);
        std::cout << "  [ok] " << path << "  (" << imgPts.size() << " corner points)\n";

        if (showImages) {
            cv::Mat vis;
            cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
            cv::aruco2::drawGridBoard(vis, board, cv::Scalar(0, 255, 0), true);
            //resize to 1280x720
            cv::resize(vis,vis,cv::Size(1280,720));
            cv::imshow("calibration", vis);
            cv::waitKey(200);
        }
    }

    if (showImages) cv::destroyAllWindows();

    if ((int)allImgPts.size() < 3) {
        std::cerr << "Need at least 3 valid images (got " << allImgPts.size() << ")\n";
        return 1;
    }
    std::cout << "\nCalibrating with " << allImgPts.size() << " images...\n";

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rpe = cv::calibrateCamera(allObjPts, allImgPts, imageSize,
                                     cameraMatrix, distCoeffs, rvecs, tvecs);

    std::cout << "Reprojection error : " << rpe << " px\n";
    std::cout << "Camera matrix:\n"       << cameraMatrix << "\n";
    std::cout << "Distortion coeffs:\n"   << distCoeffs << "\n";

    cv::FileStorage fs_out(outFile, cv::FileStorage::WRITE);
    fs_out << "image_size"          << imageSize;
    fs_out << "camera_matrix"       << cameraMatrix;
    fs_out << "distortion_coeffs"   << distCoeffs;
    fs_out << "reprojection_error"  << rpe;
    fs_out << "num_images"          << (int)allImgPts.size();
    fs_out << "board_cols"          << bw;
    fs_out << "board_rows"          << bh;
    fs_out << "marker_size_m"       << markerSize;
    fs_out << "dictionary_id"       << dictId;
    fs_out.release();

    std::cout << "Calibration saved to: " << outFile << "\n";
    std::cout << "\nTo detect markers using this calibration:\n"
              << "  aruco2_detect_markers <image_folder>"
              << " -calib=" << outFile
              << " -ms=?"   <<std::endl;
    return 0;
}
