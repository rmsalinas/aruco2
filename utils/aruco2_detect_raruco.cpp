// Detect RAruco markers in a single image, folder of images, or video file.
//
// Usage:
//   aruco2_detect_raruco <input_path> [-dict=17] [-show=true] [-save=<output_path>] [-fps=2.0]
//
#include "opencv2/objdetect/aruco2.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <deque>
#include <numeric>

namespace fs = std::filesystem;

class TimeTracker {
private:
    std::deque<double> history;
    const size_t max_size = 30;
public:
    void add(double ms) {
        history.push_back(ms);
        if (history.size() > max_size) {
            history.pop_front();
        }
    }
    double getAverage() const {
        if (history.empty()) return 0.0;
        double sum = std::accumulate(history.begin(), history.end(), 0.0);
        return sum / history.size();
    }
};

bool isImageFile(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff" || ext == ".webp";
}

bool isVideoFile(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov" || ext == ".webm" || ext == ".mpeg" || ext == ".mpg";
}

void processImage(const fs::path& inputPath, cv::aruco2::DictionaryType dict, bool show, const std::string& savePath) {
    cv::Mat image = cv::imread(inputPath.string());
    if (image.empty()) {
        std::cerr << "Error: Cannot read image: " << inputPath << "\n";
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto markers = cv::aruco2::detectRArucoMarkers(image, dict);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    cv::aruco2::drawFiducialMarkers(image, markers, cv::Scalar(255, 0, 0));

    std::cout << inputPath.filename().string() << " : " << markers.size()
              << " RAruco marker(s) detected (time: " << elapsedMs
              << " ms, avg_30: " << elapsedMs << " ms)\n";

    if (!savePath.empty()) {
        cv::imwrite(savePath, image);
    }

    if (show) {
        cv::Mat display = image;
        if (display.cols > 1280 || display.rows > 720) {
            double scale = std::min(1280.0 / display.cols, 720.0 / display.rows);
            cv::resize(display, display, cv::Size(), scale, scale);
        }
        cv::imshow("aruco2_detect_raruco", display);
        cv::waitKey(0);
    }
}

void processDirectory(const fs::path& dirPath, cv::aruco2::DictionaryType dict, bool show, const std::string& savePath, double outputFps) {
    std::vector<fs::path> paths;
    for (auto& entry : fs::directory_iterator(dirPath)) {
        if (isImageFile(entry.path())) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());

    if (paths.empty()) {
        std::cerr << "No images found in directory: " << dirPath << "\n";
        return;
    }

    cv::VideoWriter writer;
    bool writeAsVideo = false;
    cv::Size videoSize;

    if (!savePath.empty()) {
        fs::path sPath(savePath);
        std::string ext = sPath.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        writeAsVideo = (ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov");

        if (writeAsVideo) {
            cv::Mat firstImg = cv::imread(paths[0].string());
            if (!firstImg.empty()) {
                videoSize = firstImg.size();
                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                if (ext == ".avi") fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                writer.open(savePath, fourcc, outputFps, videoSize);
                if (!writer.isOpened()) {
                    std::cerr << "Warning: Cannot open VideoWriter for path: " << savePath << ". Saving output disabled.\n";
                    writeAsVideo = false;
                }
            } else {
                std::cerr << "Error: Cannot read first image to initialize VideoWriter.\n";
                writeAsVideo = false;
            }
        } else {
            fs::create_directories(savePath);
        }
    }

    std::cout << "Found " << paths.size() << " images in " << dirPath << "\n";
    if (writeAsVideo) {
        std::cout << "Saving output as video: " << savePath << " (" << videoSize.width << "x" << videoSize.height << " @ " << outputFps << " fps)\n";
    }

    TimeTracker tracker;
    bool quit = false;
    for (const auto& path : paths) {
        cv::Mat image = cv::imread(path.string());
        if (image.empty()) {
            std::cerr << "  [warn] cannot read: " << path << "\n";
            continue;
        }

        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        cv::UMat u_gray;
        std::vector<cv::aruco2::FiducialMarker> markers;
        auto t_start = std::chrono::high_resolution_clock::now();
        if (cv::ocl::useOpenCL()) {
            gray.copyTo(u_gray); // Transfer img to GPU
            markers = cv::aruco2::detectRArucoMarkers(u_gray, dict);
        } else {
            markers = cv::aruco2::detectRArucoMarkers(gray, dict);
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        tracker.add(elapsedMs);

        cv::aruco2::drawFiducialMarkers(image, markers, cv::Scalar(255, 255, 0));

        std::cout << path.filename().string() << " : " << markers.size()
                  << " RAruco marker(s) detected (time: " << elapsedMs
                  << " ms, avg_30: " << tracker.getAverage() << " ms)\n";

        if (writeAsVideo && writer.isOpened()) {
            cv::Mat frameToWrite = image;
            if (frameToWrite.size() != videoSize) {
                cv::resize(frameToWrite, frameToWrite, videoSize);
            }
            writer.write(frameToWrite);
        } else if (!savePath.empty()) {
            std::string out = (fs::path(savePath) / path.filename()).string();
            cv::imwrite(out, image);
        }

        if (show) {
            cv::Mat display = image;
            if (display.cols > 1280 || display.rows > 720) {
                double scale = std::min(1280.0 / display.cols, 720.0 / display.rows);
                cv::resize(display, display, cv::Size(), scale, scale);
            }
            cv::imshow("aruco2_detect_raruco", display);
            int key = cv::waitKey(0);
            if (key == 27) { // ESC
                quit = true;
                break;
            }
        }
    }
    if (show) {
        cv::destroyAllWindows();
    }
}

void processVideo(const fs::path& videoPath, cv::aruco2::DictionaryType dict, bool show, const std::string& savePath, double outputFps) {
    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << videoPath << "\n";
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) fps = outputFps;
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer;
    if (!savePath.empty()) {
        fs::path sPath(savePath);
        std::string ext = sPath.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        bool writeAsVideo = (ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov");

        if (writeAsVideo) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            if (ext == ".avi") fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            writer.open(savePath, fourcc, fps, cv::Size(width, height));
            if (!writer.isOpened()) {
                std::cerr << "Warning: Cannot open VideoWriter for path: " << savePath << ". Saving output disabled.\n";
            }
        } else {
            fs::create_directories(savePath);
        }
    }

    std::cout << "Processing video: " << videoPath << " (" << width << "x" << height << " @ " << fps << " fps)\n";
    if (writer.isOpened()) {
        std::cout << "Saving output as video: " << savePath << "\n";
    }

    cv::Mat frame;
    int frameIdx = 0;
    bool quit = false;
    TimeTracker tracker;
    cv::aruco2::DetectionParameters dparams;
    dparams.minSize=20;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        cv::Mat gray;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame;
        }

        cv::UMat u_gray;
        std::vector<cv::aruco2::FiducialMarker> markers;
        auto t_start = std::chrono::high_resolution_clock::now();
        if (cv::ocl::useOpenCL()) {
            gray.copyTo(u_gray); // Transfer img to GPU
            markers = cv::aruco2::detectRArucoMarkers(u_gray, dict);
        } else {
            markers = cv::aruco2::detectRArucoMarkers(gray, dict);
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        tracker.add(elapsedMs);

        cv::aruco2::drawFiducialMarkers(frame, markers, cv::Scalar(255, 255, 0));

        std::cout << "Frame " << frameIdx << " : " << markers.size()
                  << " RAruco marker(s) detected (time: " << elapsedMs
                  << " ms, avg_30: " << tracker.getAverage() << " ms)\n";

        if (writer.isOpened()) {
            writer.write(frame);
        } else if (!savePath.empty()) {
            char filename[128];
            snprintf(filename, sizeof(filename), "frame_%06d.png", frameIdx);
            std::string out = (fs::path(savePath) / filename).string();
            cv::imwrite(out, frame);
        }

        if (show) {
            cv::Mat display = frame;
            if (display.cols > 1280 || display.rows > 720) {
                double scale = std::min(1280.0 / display.cols, 720.0 / display.rows);
                cv::resize(display, display, cv::Size(), scale, scale);
            }
            cv::imshow("aruco2_detect_raruco", display);
            int delay = static_cast<int>(1000.0 / fps);
            if (delay <= 0) delay = 1;
            int key = cv::waitKey(delay);
            if (key == 27) { // ESC
                quit = true;
                break;
            }
            if(key=='w'){
                //write to a file in full res
               cv::imwrite((fs::path(savePath) / ("frame_" + std::to_string(frameIdx) + ".png")).string(), frame);
                std::cout << "Saved frame " << frameIdx << " to file.\n";
            }
        }
        frameIdx++;
    }

    std::cout << "Finished processing " << frameIdx << " frames.\n";
    if (show) {
        cv::destroyAllWindows();
    }
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv,
        "{@path  |    | image file, directory, or video file }"
        "{dict   | 17 | dictionary id (17 = DICT_APRILTAG_16h5) }"
        "{show   | true| show each result in a window (any key/next frame, ESC = quit) }"
        "{save   |    | file path (for single image/video) or folder (for directory) to write annotated output }"
        "{fps    | 25.0| frame rate for output video (used if saving directory to video or input video lacks frame rate metadata) }"
        "{help   |    | show this help message }");

    if (parser.has("help")) { parser.printMessage(); return 0; }
    if (!parser.check())    { parser.printErrors();  return 1; }

    std::string inputPathStr = parser.get<std::string>("@path");
    int dictId = parser.get<int>("dict");
    bool show = parser.get<bool>("show");
    std::string savePath = parser.get<std::string>("save");
    double outputFps = parser.get<double>("fps");

    if (inputPathStr.empty()) { parser.printMessage(); return 1; }

    fs::path inputPath(inputPathStr);
    if (!fs::exists(inputPath)) {
        std::cerr << "Error: Input path does not exist: " << inputPathStr << "\n";
        return 1;
    }

    auto dict = static_cast<cv::aruco2::DictionaryType>(dictId);

    if (fs::is_directory(inputPath)) {
        processDirectory(inputPath, dict, show, savePath, outputFps);
    } else if (isImageFile(inputPath)) {
        processImage(inputPath, dict, show, savePath);
    } else if (isVideoFile(inputPath)) {
        processVideo(inputPath, dict, show, savePath, outputFps);
    } else {
        cv::Mat testImg = cv::imread(inputPath.string());
        if (!testImg.empty()) {
            processImage(inputPath, dict, show, savePath);
        } else {
            cv::VideoCapture cap(inputPath.string());
            if (cap.isOpened()) {
                cap.release();
                processVideo(inputPath, dict, show, savePath, outputFps);
            } else {
                std::cerr << "Error: Unrecognized file type or cannot open: " << inputPathStr << "\n";
                return 1;
            }
        }
    }

    return 0;
}
