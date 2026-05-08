# aruco2 — A Simpler ArUco for OpenCV 5

<img src="types.jpg" width="75%"/>

A proposed replacement for the ArUco module in OpenCV 5, by the original ArUco author.

- **6.5× faster** detection engine based on [ArUco Nano](https://github.com/rmsalinas/aruco_nano)
- **Simpler API** — one function call, results in a single `vector<Marker>` (no parallel vectors)
- **Single public header** — `#include "aruco2.hpp"` is all you need; no extra headers to hunt down
- **Safer defaults** — `errorCorrectionRate=0` instead of the legacy 0.6 that causes false positives
- **Multi-dictionary** detection in one pass
- **Boards and diamonds** based on [ChArUco2](https://github.com/rmsalinas/charuco2) — double the marker density, twice the corners at 75% occlusion
- **Fractal markers** — nested multi-scale design gives many more corners for pose estimation, robust to heavy occlusion
- **OpenCL acceleration** *(coming soon)*

---

## Motivation

Detecting markers with the current OpenCV ArUco API looks like this:

```cpp
// Current OpenCV aruco — a lot of boilerplate for a simple task
cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
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
| Public API surface | multiple headers | single `aruco2.hpp` |
| Result type | two parallel vectors (`corners`, `ids`) | `vector<Marker>` — id and corners travel together |
| Multi-dictionary | not supported in one call | `detectMarkers(image, {DICT_A, DICT_B})` |
| Default error correction | 0.6 (causes false positives) | 0 (strict — raise only when needed) |
| Board design | markers on half the squares | ChArUco2: markers on every square |
| Diamond | separate concept using corner lists | first-class `Diamond` type with `vector<Marker>` |
| Fractal markers | not supported | first-class `FractalMarker` type — nested design, many more pose corners |
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
    DictionaryType dict;
    std::vector<Marker> markers;    // the 4 detected markers forming the diamond
};

struct FractalMarker {
    std::vector<cv::Point2f> corners; // 4 outer corners, clockwise from top-left
    FractalType type;                 // fractal configuration (FRACTAL_2L_6 … FRACTAL_5L_6)
    int id = -1;                      // id of the outer marker
};
```

---

## API

| Function | Description |
|---|---|
| `generateMarkerImage(img, dict, id)` | render a single marker to an image |
| `generateBoardImage(img, size, dict)` | render a grid board to an image |
| `generateDiamondImage(img, dict, ids)` | render a diamond (2×2 block) to an image |
| `generateFractalImage(img, ftype)` | render a fractal marker to an image |
| `detectMarkers(image, dict)` → `vector<Marker>` | find markers in an image |
| `detectMarkers(image, {dict, …})` → `vector<Marker>` | find markers across multiple dictionaries |
| `detectBoard(image, size, dict, board)` → `bool` | find a grid board |
| `detectDiamonds(image, dict)` → `vector<Diamond>` | find diamond markers |
| `detectFractals(image, ftype)` → `vector<FractalMarker>` | find fractal markers |
| `drawDetectedMarkers(image, markers)` | draw marker outlines and ids |
| `drawDetectedBoard(image, board)` | draw detected board corners |
| `drawDetectedDiamonds(image, diamonds)` | draw diamond outlines and ids |
| `drawDetectedFractals(image, fractals)` | draw fractal marker outlines, ids, and all matched image points (circles) |
| `getSolvePnpPoints(marker, imgPts, objPts)` | extract solvePnP inputs for a marker |
| `getSolvePnpPoints(board, imgPts, objPts)` | extract solvePnP inputs for a board |
| `getSolvePnpPoints(diamond, imgPts, objPts)` | extract solvePnP inputs for a diamond |
| `getSolvePnpPoints(fractal, imgPts, objPts)` | extract solvePnP inputs for a fractal marker |

Generate → Detect → Draw → Pose: four verbs, four target types, one consistent pattern.

---

## Examples

### Detect markers (single dictionary)

```cpp
#include "aruco2.hpp"

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
cv::aruco2::generateMarkerImage(markerImg, cv::aruco2::DICT_6X6_250, 42);
cv::imwrite("marker_42.png", markerImg);
```

The optional fourth argument controls bit size in pixels (default 20). A white outer border is added by default (`externalBorder=true`).

---

### Generate a diamond image

```cpp
cv::Mat diamondImg;
cv::aruco2::generateDiamondImage(diamondImg, cv::aruco2::DICT_6X6_250, {10, 11, 12, 13});
cv::imwrite("diamond.png", diamondImg);
```

The four ids are arranged clockwise from the top-left, matching the `Diamond::id` field
returned by `detectDiamonds()`.

---

### Pose estimation from a single marker

`getSolvePnpPoints` takes an optional `markerSize` parameter (physical side length, e.g. in metres).
Pass it directly — no manual scaling needed:

```cpp
cv::Mat cameraMatrix, distCoeffs; // from calibration

for (const auto &m : cv::aruco2::detectMarkers(image)) {
    cv::Mat imgPts, objPts, rvec, tvec;
    cv::aruco2::getSolvePnpPoints(m, imgPts, objPts, 0.05f); // 5 cm marker
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

    // Draw detected corners
    cv::aruco2::drawDetectedBoard(image, board);

    // Pose estimation — uses all detected board corners, robust to partial occlusion
    cv::Mat imgPts, objPts, rvec, tvec;
    cv::aruco2::getSolvePnpPoints(board, imgPts, objPts, 0.05f); // 5 cm markers
    cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec);
}
```

In Python:

```python
found, board = cv.aruco2.detectBoard(image, (4, 3), cv.aruco2.DICT_6X6_250)
```

---

### Diamond detection and pose estimation

A diamond is a 2×2 block of markers.  Its identity is the combination of the four constituent marker ids, stored as a `Vec4i`.  `getSolvePnpPoints` returns the full 9-point 3×3 corner grid of the diamond.

```cpp
auto diamonds = cv::aruco2::detectDiamonds(image, cv::aruco2::DICT_6X6_250);

// Draw detected diamonds
cv::aruco2::drawDetectedDiamonds(image, diamonds);

for (const auto &d : diamonds) {
    std::cout << "Diamond ids: "
              << d.id[0] << " " << d.id[1] << " "
              << d.id[2] << " " << d.id[3] << "\n";

    // imgPts and objPts are 9×1 (full 3×3 corner grid)
    cv::Mat imgPts, objPts, rvec, tvec;
    cv::aruco2::getSolvePnpPoints(d, imgPts, objPts, 0.05f); // 5 cm markers
    cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec);
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

### Fractal marker generation and detection

A fractal marker is a nested design: the outer 6×6 marker contains one or more smaller
markers at decreasing scales.  More levels mean more visible corners, giving better pose
accuracy and robustness to partial occlusion.

```cpp
// Generate a FRACTAL_3L_6 image (3 nesting levels)
cv::Mat fractalImg;
cv::aruco2::generateFractalImage(fractalImg, cv::aruco2::FRACTAL_3L_6);
cv::imwrite("fractal.png", fractalImg);

// Detect fractal markers
auto fractals = cv::aruco2::detectFractals(image, cv::aruco2::FRACTAL_3L_6);

// Draw results — outline, id, and all matched image points (default)
cv::aruco2::drawDetectedFractals(image, fractals);

// Outline and id only, no image points
// cv::aruco2::drawDetectedFractals(image, fractals, cv::Scalar(0,255,0), false);

for (const auto &f : fractals)
    std::cout << "id=" << f.id << " corner0=" << f.corners[0] << "\n";
```

### Fractal marker pose estimation

When only one fractal marker is visible, `getSolvePnpPoints` returns all inner and outer
corners — far more correspondences than a plain marker provides:

```cpp
cv::Mat cameraMatrix, distCoeffs; // from calibration

auto fractals = cv::aruco2::detectFractals(image, cv::aruco2::FRACTAL_3L_6);

for (const auto &f : fractals) {
    cv::Mat imgPts, objPts, rvec, tvec;
    cv::aruco2::getSolvePnpPoints(f, imgPts, objPts, 0.10f); // 10 cm outer marker
    cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec);
}
```

---

## Implementation

### Marker detection — based on ArUco Nano

The marker detector is based on [ArUco Nano](https://github.com/rmsalinas/aruco_nano), a
high-performance single-header detector described in:

> R. Muñoz-Salinas et al., *"ArUco Nano: a simpler, faster, and more reliable fiducial marker
> detector"*, SoftwareX, 2026.

Key advantages over the standard OpenCV ArUco detector:

- **Up to 6.5× faster** than OpenCV ArUco single-threaded, **2× faster** than its
  multi-threaded mode (benchmarked on Intel Core i7-13700H)

| Resolution | aruco2 | OpenCV ArUco | Speedup |
|---|---|---|---|
| 1 MP  |   6.68 ms |  43.46 ms | 6.5× |
| 4 MP  |  22.93 ms | 125.00 ms | 5.4× |
| 16 MP | 101.67 ms | 504.27 ms | 4.9× |

- **Visited-aware contour tracer** — prunes revisited pixels to suppress noise and thin
  structures, reducing false candidates before any dictionary lookup
- **SIMD-accelerated** via OpenCV universal intrinsics
- **Multi-attempt corner perturbation** — slightly jitters corners on retry attempts to
  improve robustness under perspective distortion

---

### Boards and diamonds — based on ChArUco2

The board and diamond design is based on [ChArUco2](https://github.com/rmsalinas/charuco2),
described in:

> R. Muñoz-Salinas et al., *"ChArUco2: Enhanced Calibration Boards with Dual Black-and-White
> Marker Detection"*, SoftwareX, submitted.

Key advantages over standard OpenCV ChArUco:

- **Double marker density** — every square carries a marker (standard on black, inverted on
  white) versus ~half the squares in standard ChArUco
- **Larger markers** — each occupies the full square area with no white border, improving
  detection range under challenging lighting
- **More reference corners** — (N+1)×(M+1) observable corners including the board border,
  versus (N−1)×(M−1) inner corners in standard ChArUco
- **Better diamond** — the 2×2 marker block provides more observations and a larger geometric
  baseline than the standard 4-corner diamond

| Occlusion | ChArUco corners | ChArUco2 corners |
|---|---|---|
| 0%  | 32 |  60 |
| 25% | 24 |  48 |
| 50% | 16 |  36 |
| 75% |  4 |  12 |

*(9×5 board, 26 mm squares, DICT_ARUCO_MIP_36h12)*

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
| Generate board images | done |
| Generate diamond images | done |
| Board detection | done |
| Draw detected board | done |
| Diamond detection | done |
| Draw detected diamonds | done |
| Pose estimation — single marker | done |
| Pose estimation — board | done |
| Pose estimation — diamond | done |
| Generate fractal marker images | done |
| Fractal marker detection | done |
| Draw detected fractal markers | done |
| Pose estimation — fractal marker | done |
| Python bindings | designed, pending OpenCV integration |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
