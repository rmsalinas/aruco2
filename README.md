# aruco2 — A Simpler ArUco for OpenCV 5

A proposed replacement for the ArUco module in OpenCV 5.  
The existing implementation works, but its API has accumulated complexity over the years that makes common tasks harder than they need to be.  This project explores a cleaner design.

---

## Motivation

Detecting markers with the current OpenCV ArUco API looks like this:

```cpp
// Current OpenCV aruco — a lot of boilerplate for a simple task
cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::aruco::DetectorParameters params;
params.errorCorrectionRate = 0.6; // dangerously high default — causes false positives
cv::aruco::ArucoDetector detector(dict, params);

std::vector<std::vector<cv::Point2f>> corners;
std::vector<int> ids;
std::vector<std::vector<cv::Point2f>> rejected;
detector.detectMarkers(image, corners, ids, rejected);

// corners and ids are parallel vectors — easy to get out of sync
for (size_t i = 0; i < ids.size(); i++)
    std::cout << "id=" << ids[i] << " corner0=" << corners[i][0] << "\n";
```

With `aruco2` the same task is a single line:

```cpp
// aruco2 — detect and iterate
for (auto &marker : cv::aruco2::detectMarkers(image))
    std::cout << "id=" << marker.id << " corner0=" << marker.corners[0] << "\n";
```

---

## Key design changes

| | Current OpenCV aruco | aruco2 |
|---|---|---|
| Entry point | `ArucoDetector` class instance | free function `detectMarkers()` |
| Result type | two parallel vectors (`corners`, `ids`) | `vector<Marker>` — id and corners travel together |
| Multi-dictionary | not supported in one call | `detectMarkers(image, {DICT_A, DICT_B})` |
| Default error correction | 0.6 (causes false positives) | 0 (strict — raise only when needed) |
| Board design | markers on half the squares | ChArUco2: markers on every square |
| Diamond | separate concept using corner lists | first-class `Diamond` type with `vector<Marker>` |
| Python result | separate lists | `markers = cv.aruco2.detectMarkers(image)` |

---

## Data types

```cpp
struct Marker {
    std::vector<cv::Point2f> corners; // 4 corners, clockwise from top-left
    int id = -1;
    DictionaryType dict;
};

struct Board {
    cv::Size gridSize;              // columns × rows
    DictionaryType dict;
    std::vector<Marker> markers;    // detected markers (subset when partially occluded)
};

struct Diamond {
    cv::Vec4i id;                   // ids of the 4 constituent markers (clockwise from top-left)
    std::vector<Marker> markers;    // the 4 detected markers forming the diamond
    DictionaryType dict;
};
```

---

## Examples

### Detect markers (single dictionary)

```cpp
#include "aruco.hpp"

cv::Mat image = cv::imread("scene.jpg");

auto markers = cv::aruco2::detectMarkers(image);  // default: DICT_ARUCO_MIP_36h12

for (const auto &m : markers)
    std::cout << "id=" << m.id << " corner0=" << m.corners[0] << "\n";
```

Pass a different dictionary as the second argument:

```cpp
auto markers = cv::aruco2::detectMarkers(image, cv::aruco2::DICT_6X6_250);
```

---

### Detect markers across multiple dictionaries

```cpp
using namespace cv::aruco2;

auto markers = detectMarkers(image, {DICT_6X6_250, DICT_APRILTAG_36h11});

for (const auto &m : markers) {
    std::string dictName = (m.dict == DICT_6X6_250) ? "aruco" : "apriltag";
    std::cout << dictName << " id=" << m.id << "\n";
}
```

Each candidate is matched against all dictionaries in a single pass and carries the dictionary it was found in.

---

### Draw detected markers

```cpp
auto markers = cv::aruco2::detectMarkers(image);
cv::aruco2::drawDetectedMarkers(image, markers);                          // green border
cv::aruco2::drawDetectedMarkers(image, markers, cv::Scalar(255, 0, 0));   // blue border
cv::imshow("markers", image);
```

Each marker is drawn with a coloured outline, the id at its centre, and a dot on `corners[0]` to show orientation.

---

### Generate a marker image

```cpp
cv::Mat markerImg;
cv::aruco2::generateImageMarker(cv::aruco2::DICT_6X6_250, 42, 300, markerImg);
cv::imwrite("marker_42.png", markerImg);
```

---

### Pose estimation from a single marker

`getSolvePnpPoints` returns object points with unit side length.  
Scale by the physical marker size before calling `solvePnP`:

```cpp
double markerSideMeters = 0.05; // 5 cm
cv::Mat cameraMatrix, distCoeffs; // from calibration

for (const auto &m : cv::aruco2::detectMarkers(image)) {
    cv::Mat imgPts, objPts, rvec, tvec;
    cv::aruco2::getSolvePnpPoints(m, imgPts, objPts);
    objPts *= markerSideMeters;
    cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec);
}
```

---

### Board generation and detection

Boards follow the [ChArUco2](https://github.com/rmsalinas/charuco2) design: every square carries a marker — standard markers on black squares and inverted markers on white squares — doubling the marker density compared to standard ChArUco.

```cpp
// Generate a 4×3 board image
cv::Mat boardImg;
cv::aruco2::generateBoardImage(boardImg, cv::Size(4, 3), cv::aruco2::DICT_6X6_250);
cv::imwrite("board.png", boardImg);

// Detect the board
cv::aruco2::Board board;
bool found = cv::aruco2::detectBoard(image, cv::Size(4, 3),
                                     cv::aruco2::DICT_6X6_250, board);
if (found) {
    std::cout << "Detected " << board.markers.size() << " of 12 markers\n";

    cv::Mat imgPts, objPts;
    cv::aruco2::getSolvePnpPoints(board, imgPts, objPts);
    objPts *= markerSideMeters;
    // pass to solvePnP as usual
}
```

In Python:

```python
found, board = cv.aruco2.detectBoard(image, (4, 3), cv.aruco2.DICT_6X6_250)
```

---

### Diamond detection and pose estimation

A diamond is a 2×2 block of markers.  Its identity is the combination of the four constituent marker ids, stored as a `Vec4i`.

```cpp
auto diamonds = cv::aruco2::detectDiamonds(image, cv::aruco2::DICT_6X6_250);

for (const auto &d : diamonds) {
    std::cout << "Diamond ids: "
              << d.id[0] << " " << d.id[1] << " "
              << d.id[2] << " " << d.id[3] << "\n";

    cv::Mat imgPts, objPts;
    cv::aruco2::getSolvePnpPoints(d, imgPts, objPts);
    objPts *= markerSideMeters;
    // pass to solvePnP as usual
}
```

---

### Tuning detection

The defaults are intentionally strict to avoid false positives.  Relax them only as needed:

```cpp
cv::aruco2::DetectorParameters params;
params.errorCorrectionRate         = 0.5;  // tolerate some bit errors
params.maxErroneousBitsInBorderRate = 0.05; // tolerate slight border damage
params.detectInvertedMarker        = true;  // white markers on black background

auto markers = cv::aruco2::detectMarkers(image, cv::aruco2::DICT_6X6_250, params);
```

---

## Building

Requires OpenCV 4 or 5.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## Status

| Feature | Status |
|---|---|
| Marker detection | done |
| Multi-dictionary detection | done |
| Draw detected markers | done |
| Generate marker images | done |
| Pose estimation — single marker | done |
| Generate board images | done |
| Board detection | in progress |
| Pose estimation — board | in progress |
| Diamond detection | in progress |
| Pose estimation — diamond | in progress |
| Python bindings | designed, pending OpenCV integration |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
