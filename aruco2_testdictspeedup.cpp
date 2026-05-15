#include "opencv2/objdetect/aruco2.hpp"
#include "aruco_dictionary.hpp"
#include <opencv2/objdetect/aruco_dictionary.hpp> // Official OpenCV ArUco
#include <opencv2/core.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace cv;
using namespace std;

struct BenchResult {
    string dictName;
    double aruco2Exact;
    double officialExact;
    double aruco2Corr;
    double officialCorr;
};

int main() {
    vector<int> dicts = {cv::aruco2::DICT_6X6_250, cv::aruco2::DICT_APRILTAG_36h10};
    vector<BenchResult> results;

    for (int dictId : dicts) {
        // Load dictionaries
        cv::aruco2::Dictionary dict2 = cv::aruco2::getPredefinedDictionary(static_cast<cv::aruco2::DictionaryType>(dictId));
        cv::aruco::Dictionary dictOfficial = cv::aruco::getPredefinedDictionary(static_cast<cv::aruco::PredefinedDictionaryType>(dictId));

        string name = (dictId == cv::aruco2::DICT_6X6_250 ? "DICT_6X6_250" : "DICT_APRILTAG_36h10");
        int totalMarkers = dict2.size();
        int markersToTest = std::min(totalMarkers, 20);

        cout << "\n---------------------------------------------------" << endl;
        cout << "Dictionary: " << name << " (" << totalMarkers << " markers)" << endl;
        cout << "Testing first " << markersToTest << " markers..." << endl;

        // Generate markers bits
        vector<Mat> markers;
        for (int i = 0; i < markersToTest; i++) {
            Mat m = dict2.getMarkerBits(i);
            m.convertTo(m, CV_8UC1);
            markers.push_back(m);
        }

        const int iterations = 5000; 
        int idx, rotation;
        BenchResult res;
        res.dictName = name;
        long long sum = 0;

        // --- 1. EXACT MATCH (maxCorrectionRate = 0.0) ---
        
        // aruco2
        auto start = chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < markersToTest; i++) {
                if (dict2.identify(markers[i], idx, rotation, 0.0)) {
                    sum += idx + rotation;
                }
            }
        }
        auto end = chrono::high_resolution_clock::now();
        res.aruco2Exact = (chrono::duration<double>(end - start).count() / (iterations * markersToTest)) * 1e6;

        // Official
        start = chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < markersToTest; i++) {
                if (dictOfficial.identify(markers[i], idx, rotation, 0.0)) {
                    sum += idx + rotation;
                }
            }
        }
        end = chrono::high_resolution_clock::now();
        res.officialExact = (chrono::duration<double>(end - start).count() / (iterations * markersToTest)) * 1e6;


        // --- 2. WITH ERROR CORRECTION (maxCorrectionRate = 0.2) ---
        double corrRate = 0.2;
        
        // aruco2
        start = chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < markersToTest; i++) {
                if (dict2.identify(markers[i], idx, rotation, corrRate)) {
                    sum += idx + rotation;
                }
            }
        }
        end = chrono::high_resolution_clock::now();
        res.aruco2Corr = (chrono::duration<double>(end - start).count() / (iterations * markersToTest)) * 1e6;

        // Official
        start = chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < markersToTest; i++) {
                if (dictOfficial.identify(markers[i], idx, rotation, corrRate)) {
                    sum += idx + rotation;
                }
            }
        }
        end = chrono::high_resolution_clock::now();
        res.officialCorr = (chrono::duration<double>(end - start).count() / (iterations * markersToTest)) * 1e6;

        cout << "Exact  -> aruco2: " << setw(8) << res.aruco2Exact << " us | Official: " << setw(8) << res.officialExact << " us" << " (sum=" << sum << ")" << endl;
        cout << "Corr.  -> aruco2: " << setw(8) << res.aruco2Corr  << " us | Official: " << setw(8) << res.officialCorr  << " us" << endl;

        results.push_back(res);
    }

    cout << "\n\n================ FULL COMPARISON WITH OFFICIAL OPENCV ================" << endl;
    cout << left << setw(20) << "Dictionary" 
         << setw(12) << "Mode"
         << setw(15) << "aruco2 (us)" 
         << setw(15) << "Official (us)" 
         << setw(10) << "Speedup" << endl;
    cout << string(75, '-') << endl;

    for (const auto& r : results) {
        cout << left << setw(20) << r.dictName 
             << setw(12) << "Exact"
             << setw(15) << fixed << setprecision(2) << r.aruco2Exact 
             << setw(15) << r.officialExact 
             << setw(10) << (to_string(int(r.officialExact / r.aruco2Exact)) + "x") << endl;
        
        cout << left << setw(20) << "" 
             << setw(12) << "Correction"
             << setw(15) << r.aruco2Corr 
             << setw(15) << r.officialCorr 
             << setw(10) << (to_string(int(r.officialCorr / r.aruco2Corr)) + "x") << endl;
        cout << string(75, '-') << endl;
    }
    cout << "======================================================================" << endl;

    return 0;
}
