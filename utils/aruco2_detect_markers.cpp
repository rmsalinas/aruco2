// Detect ArUco2 markers in every image of a folder.
// Optionally runs solvePnP and draws coordinate axes when a calibration file is given.
//
// Usage:
//   aruco2_detect_markers <image_folder> [-dict=21] [-calib=calibration.yaml] [-ms=0.05]
//                         [-show=true] [-save=<output_folder>]
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
        "{@path  |    | folder with images (jpg/png) }"
        "{dict   | 21 | dictionary id (21 = DICT_ARUCO_MIP_36h12) }"
        "{calib  |    | calibration YAML/XML produced by aruco2_camera_calibration }"
        "{ms     | 0.05| physical marker side length in metres (used for pose) }"
        "{show   | true| show each result in a window (any key = next, ESC = quit) }"
        "{save   |    | folder to write annotated images into }"
        "{help   |    | show this help message }");

    if (parser.has("help")) { parser.printMessage(); return 0; }
    if (!parser.check())    { parser.printErrors();  return 1; }

    std::string folder   = parser.get<std::string>("@path");
    std::string calibFile= parser.get<std::string>("calib");
    bool        show     = parser.get<bool>("show");
    std::string saveDir  = parser.get<std::string>("save");

    // dict and ms may be overridden by values stored in the calibration file
    int   dictId = parser.get<int>("dict");
    float ms     = parser.get<float>("ms");

    if (folder.empty()) { parser.printMessage(); return 1; }

    // --- load calibration if provided ---
    cv::Mat cameraMatrix, distCoeffs;
    bool hasCalib = false;
    if (!calibFile.empty()) {
        cv::FileStorage fs_in(calibFile, cv::FileStorage::READ);
        if (!fs_in.isOpened()) {
            std::cerr << "Cannot open calibration file: " << calibFile << "\n";
            return 1;
        }
        fs_in["camera_matrix"]     >> cameraMatrix;
        fs_in["distortion_coeffs"] >> distCoeffs;
        // use stored dict/ms as defaults unless user explicitly passed them
        if (!parser.has("dict") && !fs_in["dictionary_id"].empty())
            fs_in["dictionary_id"] >> dictId;
        if (!parser.has("ms") && !fs_in["marker_size_m"].empty())
            fs_in["marker_size_m"] >> ms;
        fs_in.release();
        if (cameraMatrix.empty()) {
            std::cerr << "calibration file missing 'camera_matrix'\n";
            return 1;
        }
        hasCalib = true;
        std::cout << "Loaded calibration from: " << calibFile
                  << "  (dict=" << dictId << ", ms=" << ms << ")\n";
    }

    auto dict = static_cast<cv::aruco2::DictionaryType>(dictId);

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

    if (!saveDir.empty())
        fs::create_directories(saveDir);

    std::cout << "Found " << paths.size() << " images, dict=" << dictId
              << (hasCalib ? ", pose estimation ON" : "") << "\n";

    bool quit = false;
    for (auto& path : paths) {
        cv::Mat image = cv::imread(path);
        if (image.empty()) { std::cerr << "  [warn] cannot read: " << path << "\n"; continue; }

        auto markers = cv::aruco2::detectMarkers(image, dict);
        cv::aruco2::drawDetected(image, markers);

        std::cout << fs::path(path).filename().string()
                  << " : " << markers.size() << " marker(s) detected";

        if (hasCalib && !markers.empty()) {
            for (const auto& marker : markers) {
                cv::Mat imgPts, objPts, rvec, tvec;
                cv::aruco2::getSolvePnpPoints(marker, objPts, imgPts, ms);
                cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec);
                cv::aruco2::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, ms * 0.5f);
            }
            std::cout << "  [pose drawn]";
        }
        std::cout << "\n";

        if (!saveDir.empty()) {
            std::string out = saveDir + "/" + fs::path(path).filename().string();
            cv::imwrite(out, image);
        }

        if (show) {
            cv::Mat display = image;
            if (display.cols > 1280 || display.rows > 720) {
                double scale = std::min(1280.0 / display.cols, 720.0 / display.rows);
                cv::resize(display, display, cv::Size(), scale, scale);
            }
            cv::imshow("aruco2_detect_markers", display);
            int key = cv::waitKey(0);
            if (key == 27) { quit = true; break; } // ESC
        }
    }

    if (show) cv::destroyAllWindows();
    return quit ? 0 : 0;
}
