#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco2.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    CommandLineParser parser(argc, argv,
        "{@dataset | | Path to the dataset directory}"
        "{@type    | | Type of marker (marker, fractal, raruco)}"
        "{@out     | | Output CSV file path}"
    );

    if (!parser.has("@dataset") || !parser.has("@type") || !parser.has("@out")) {
        parser.printMessage();
        return 0;
    }

    String datasetPath = parser.get<String>("@dataset");
    String type = parser.get<String>("@type");
    String outPath = parser.get<String>("@out");

    if (type != "marker" && type != "fractal" && type != "raruco") {
        cerr << "Invalid type. Must be marker, fractal, or raruco." << endl;
        return -1;
    }

    vector<String> filenames;
    glob(datasetPath + "/*.jpg", filenames, false);

    if (filenames.empty()) {
        cerr << "No .jpg files found in " << datasetPath << endl;
        return -1;
    }

    ofstream out(outPath);
    if (!out.is_open()) {
        cerr << "Could not open output file " << outPath << endl;
        return -1;
    }

    out << "filename,distance,angle,is_detected,time_ms\n";

    aruco2::DetectionParameters params;

    int count = 0;
    for (const auto& fpath : filenames) {
        size_t lastSlash = fpath.find_last_of("/\\");
        String fname = (lastSlash == String::npos) ? fpath : fpath.substr(lastSlash + 1);

        // Parse distance and angle from sim_CameraName_distX.X_angleY.Y.jpg
        size_t distPos = fname.find("_dist");
        size_t anglePos = fname.find("_angle");
        size_t extPos = fname.find(".jpg");
        
        if (distPos == String::npos || anglePos == String::npos || extPos == String::npos) {
            continue;
        }

        String distStr = fname.substr(distPos + 5, anglePos - (distPos + 5));
        String angleStr = fname.substr(anglePos + 6, extPos - (anglePos + 6));

        float distance = stof(distStr);
        float angle = stof(angleStr);

        Mat img = imread(fpath, IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        bool detected = false;
        
        int64 t0 = getTickCount();
        
        if (type == "marker") {
            vector<aruco2::FiducialMarker> markers = aruco2::detectFiducialMarkers(img, aruco2::DICT_ARUCO_MIP_36h12, params);
            if (!markers.empty()) detected = true;
        } else if (type == "fractal") {
            vector<aruco2::FractalMarker> fractals = aruco2::detectFractals(img, aruco2::FRACTAL_3L_6);
            if (!fractals.empty()) detected = true;
        } else if (type == "raruco") {
            vector<aruco2::FiducialMarker> markers = aruco2::detectRArucoMarkers(img, aruco2::DICT_APRILTAG_16h5, params);
            if (!markers.empty()) detected = true;
        }

        int64 t1 = getTickCount();
        double time_ms = (t1 - t0) * 1000.0 / getTickFrequency();

        out << fname << "," << distance << "," << angle << "," << (detected ? 1 : 0) << "," << time_ms << "\n";
        
        count++;
        if (count % 100 == 0) {
            cout << "Processed " << count << "/" << filenames.size() << " images." << endl;
        }
    }

    out.close();
    cout << "Finished processing " << count << " images." << endl;
    return 0;
}
