// Detect ArUco2 fractal markers in every image of a folder.
// Optionally runs solvePnP and draws coordinate axes when a calibration file is given.
// The fractal object coordinate system is centred at (0,0,0); the outer marker spans
// [-ms/2, +ms/2] on X and Y, with Z pointing toward the camera.
//
// Usage:
//   aruco2_detect_fractal <image_folder> [-ftype=2] [-calib=calibration.yaml] [-ms=0.10]
//                         [-show=true] [-save=<output_folder>]
//
// ftype: 0=FRACTAL_2L_6  1=FRACTAL_3L_6  2=FRACTAL_4L_6  3=FRACTAL_5L_6
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
        "{ftype  | 1  | fractal type: 0=2L_6 1=3L_6 2=4L_6 3=5L_6 }"
        "{calib  |    | calibration YAML/XML produced by aruco2_camera_calibration }"
        "{ms     | 0.10| physical side length of the outer marker in metres }"
        "{show   | true| show each result in a window (any key = next, ESC = quit) }"
        "{save   |    | folder to write annotated images into }"
        "{help   |    | show this help message }");

    if (parser.has("help")) { parser.printMessage(); return 0; }
    if (!parser.check())    { parser.printErrors();  return 1; }

    std::string folder   = parser.get<std::string>("@path");
    std::string calibFile= parser.get<std::string>("calib");
    bool        show     = parser.get<bool>("show");
    std::string saveDir  = parser.get<std::string>("save");

    // ms may be overridden by the calibration file; ftype has no equivalent there
    int   ftypeId = parser.get<int>("ftype");
    float ms      = parser.get<float>("ms");

    if (folder.empty()) { parser.printMessage(); return 1; }
    if (ftypeId < 0 || ftypeId > 3) {
        std::cerr << "ftype must be 0..3\n";
        return 1;
    }

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
        if (!parser.has("ms") && !fs_in["marker_size_m"].empty())
            fs_in["marker_size_m"] >> ms;
        fs_in.release();
        if (cameraMatrix.empty()) {
            std::cerr << "calibration file missing 'camera_matrix'\n";
            return 1;
        }
        hasCalib = true;
        std::cout << "Loaded calibration from: " << calibFile
                  << "  (ms=" << ms << ")\n";
    }

    auto ftype = static_cast<cv::aruco2::FractalType>(ftypeId);

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

    std::cout << "Found " << paths.size() << " images, ftype=" << ftypeId
              << (hasCalib ? ", pose estimation ON" : "") << "\n";

    bool quit = false;
    for (auto& path : paths) {
        cv::Mat image = cv::imread(path);
        if (image.empty()) { std::cerr << "  [warn] cannot read: " << path << "\n"; continue; }

        auto fractals = cv::aruco2::detectFractals(image, ftype);
        cv::aruco2::drawFractals(image, fractals);

        std::cout << fs::path(path).filename().string()
                  << " : " << fractals.size() << " fractal(s) detected";

        if (hasCalib && !fractals.empty()) {
            for (const auto& fractal : fractals) {
                cv::Mat imgPts, objPts, rvec, tvec;
                cv::aruco2::getSolvePnpPoints(fractal, objPts, imgPts, ms);
                cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec);
                // axis length = ms/2 so it reaches the outer marker edge
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
            cv::imshow("aruco2_detect_fractal", display);
            int key = cv::waitKey(0);
            if (key == 27) { quit = true; break; }
        }
    }

    if (show) cv::destroyAllWindows();
    return 0;
}
