// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#pragma once
#include <opencv2/core.hpp>
#include "aruco_dictionary.hpp"

namespace cv :: aruco2 {

//! @addtogroup objdetect_aruco
//! @{

/** @brief Detection parameters for detectMarkers() and detectBoard().
 *
 * All parameters have defaults that work well for standard printed markers under normal lighting.
 * Tune only when detection fails or produces false positives in your specific setup.
 */
struct CV_EXPORTS_W_SIMPLE DetectorParameters {

    /** @brief Size of the box filter kernel used for adaptive thresholding (pixels, must be odd).
     *
     * Larger values tolerate more uneven lighting but may merge nearby markers.
     * Default: 15.
     */
    CV_PROP_RW int boxFilterSize = 15;

    /** @brief Threshold offset applied after the box filter subtraction.
     *
     * A pixel is considered foreground if `boxFilter(p) - p > thres`.
     * Increase to suppress noise; decrease to detect faint borders.
     * Default: 3.
     */
    CV_PROP_RW int thres = 3;

    /** @brief Minimum side length (pixels) for a contour to be considered a marker candidate.
     *
     * Contour sides shorter than this are discarded early.
     * Default: 10.
     */
    CV_PROP_RW int minSize = 10;

    /** @brief Number of attempts to identify a candidate by slightly perturbing its corners.
     *
     * On each attempt after the first, Gaussian noise (σ=0.75 px) is added to the corners
     * before re-sampling the bits.  Improves robustness near perspective extremes.
     * Default: 5.
     */
    CV_PROP_RW int maxAttemptsPerCandidate = 5;

    /** @brief Controls how aggressively the contour tracer prunes revisited paths.
     *
     * Expressed as a fraction of the total contour length [0, 1].  A contour is discarded
     * if the number of already-visited pixels exceeds `maxTimesRevisited * contourLength`.
     * - 0.05 (default): filters most noise and thin structures efficiently.
     * - 1.0 : behaves like a standard Moore neighbour tracer (no pruning).
     *
     * Lower values speed up detection and reduce false candidates at the cost of occasionally
     * missing very distorted markers.
     */
    CV_PROP_RW float maxTimesRevisited = 0.05f;

    /** @brief Width of the mandatory black border around each marker, in bits.
     *
     * Almost all standard dictionaries use 1 border bit.  Kept for compatibility with
     * custom dictionaries that deviate from this convention.
     * Default: 1.
     */
    CV_PROP_RW int markerBorderBits = 1;

    /** @brief Fraction of `maxCorrectionBits` to use when matching a candidate against the dictionary.
     *
     * A candidate is accepted if its Hamming distance to the nearest dictionary entry is at most
     * `floor(maxCorrectionBits * errorCorrectionRate)`.
     * - 0 (default): no bit errors tolerated — lowest false-positive rate.
     * - 1.0: use the full error-correction capacity of the dictionary.
     *
     * @warning The legacy OpenCV ArUco default of 0.6 produces many false positives in cluttered
     * scenes.  Raise this only if you need tolerance against printing or lighting artefacts and
     * accept the trade-off.
     */
    CV_PROP_RW double errorCorrectionRate = 0;

    /** @brief Maximum fraction of border bits allowed to be wrong before rejecting a candidate.
     *
     * Set to 0 (default) to require a perfect black border.  Small non-zero values (e.g. 0.05)
     * add tolerance for border damage or ink bleed.
     */
    CV_PROP_RW double maxErroneousBitsInBorderRate = 0;

    /** @brief Set to true to detect markers printed white-on-black (inverted polarity).
     *
     * Default: false (standard black-on-white markers).
     */
    CV_PROP_RW bool detectInvertedMarker = false;
};


/** @brief A single detected ArUco marker.
 *
 * `corners` holds the four image-plane corner points in clockwise order starting from the
 * top-left corner.  `id` is the marker identifier within its `dict` dictionary.
 *
 * Corner order (viewed from front, standard orientation):
 * @code
 *   corners[0] ---- corners[1]
 *       |                |
 *   corners[3] ---- corners[2]
 * @endcode
 *
 * @sa detectMarkers, drawDetectedMarkers, getSolvePnpPoints
 */
struct CV_EXPORTS_W_SIMPLE Marker {
    CV_PROP_RW std::vector<cv::Point2f> corners; ///< four corner points in clockwise order
    CV_PROP_RW int id = -1;                      ///< marker id; -1 if unidentified
    CV_PROP_RW DictionaryType dict = DictionaryType(-1); ///< dictionary this marker belongs to
};


/** @brief Generate a canonical marker image ready for printing.
 *
 * @param dictionary  predefined dictionary the marker belongs to
 * @param id          marker identifier; must be a valid index in the chosen dictionary
 * @param sidePixels  output image size in pixels (square); must be >= markerSize + 2*borderBits
 * @param img         output grayscale image (CV_8UC1)
 * @param borderBits  width of the black border in marker bits (default 1)
 *
 * Example — save a 200×200 px image of marker 42 from the 6×6 250-code dictionary:
 * @code
 * cv::Mat markerImg;
 * cv::aruco2::generateImageMarker(DICT_6X6_250, 42, 200, markerImg);
 * cv::imwrite("marker_42.png", markerImg);
 * @endcode
 */
CV_WRAP void generateImageMarker(const DictionaryType &dictionary, int id, int sidePixels,
                                 OutputArray img, int borderBits = 1);


/** @brief Detect ArUco markers in an image using a single dictionary.
 *
 * @param image        input image (grayscale or BGR)
 * @param dict         dictionary to search; default is DICT_ARUCO_MIP_36h12
 * @param detectorParams  detection tuning parameters
 * @return             vector of detected Marker objects; empty if none found
 *
 * Performs the full detection pipeline: adaptive thresholding → contour tracing →
 * quadrilateral fitting → bit extraction → dictionary lookup → subpixel corner refinement.
 *
 * @note Lens distortion is not corrected internally.  For accurate pose estimation,
 * undistort the image first with the known camera model.
 * @sa undistort, detectMarkers(InputArray, const std::vector<DictionaryType>&, const DetectorParameters&)
 */
CV_WRAP std::vector<Marker> detectMarkers(InputArray image, DictionaryType dict = DICT_ARUCO_MIP_36h12,
                                          const DetectorParameters &detectorParams = {});

/** @brief Detect ArUco markers in an image searching across multiple dictionaries in one pass.
 *
 * @param image        input image (grayscale or BGR)
 * @param dicts        list of dictionaries to search simultaneously
 * @param detectorParams  detection tuning parameters
 * @return             vector of detected Marker objects; each carries the dictionary it was found in
 *
 * Each marker candidate is tested against all dictionaries in `dicts`.  Once identified in one
 * dictionary it is removed from the candidate pool, so the same region is never matched twice.
 *
 * @sa Marker::dict
 */
CV_WRAP std::vector<Marker> detectMarkers(InputArray image, const std::vector<DictionaryType> &dicts,
                                          const DetectorParameters &detectorParams = {});


/** @brief Draw detected markers onto an image.
 *
 * @param image        input/output image (1 or 3 channels); modified in place
 * @param markers      markers returned by detectMarkers()
 * @param borderColor  color used to draw the marker outline (default: green)
 *
 * For each marker the function draws:
 * - a quadrilateral outline in `borderColor`
 * - a filled circle on corner[0] in a contrasting color to indicate orientation
 * - the marker id as text at the marker centroid
 *
 * Useful for visualisation and debugging.
 */
CV_WRAP void drawDetectedMarkers(InputOutputArray image, const std::vector<Marker> &markers,
                                 Scalar borderColor = Scalar(0, 255, 0));


/** @brief Compute image and object points for a single marker to pass to solvePnP().
 *
 * @param marker     a detected marker
 * @param imgPoints  output 4×1 array of the marker's corner pixel coordinates (CV_32FC2)
 * @param objPoints  output 4×1 array of the corresponding 3-D object points in marker
 *                   coordinates (CV_32FC3), with the marker centre at the origin and
 *                   half-unit side length (i.e. corners at ±0.5 in X and Y, Z=0)
 *
 * Scale `objPoints` by the known physical marker side length before calling solvePnP():
 * @code
 * cv::Mat rvec, tvec, imgPts, objPts;
 * cv::aruco2::getSolvePnpPoints(marker, imgPts, objPts);
 * objPts *= markerSideMeters;
 * cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec);
 * @endcode
 */
CV_WRAP void getSolvePnpPoints(const Marker marker, OutputArray imgPoints, OutputArray objPoints);


/** @brief Result of detecting a ChArUco2-style grid board.
 *
 * Follows the ChArUco2 design: every square carries an ArUco marker (standard markers on black
 * squares, inverted markers on white squares), yielding N×M markers on an N×M board and
 * (N+1)×(M+1) observable intersection corners including the board border.
 *
 * `markers` holds the detected ArUco markers (a subset of all board markers when the board is
 * partially occluded).  Use getSolvePnpPoints() to obtain the corresponding image and object
 * point arrays for solvePnP().
 *
 * @sa detectBoard, getSolvePnpPoints(const Board, OutputArray, OutputArray)
 */
struct CV_EXPORTS_W_SIMPLE Board {
    CV_PROP_RW cv::Size gridSize;              ///< board dimensions: width × height in markers
    CV_PROP_RW DictionaryType dict;            ///< dictionary used for all markers on the board
    CV_PROP_RW std::vector<Marker> markers;    ///< detected markers (subset of the full board)
};

/** @brief Generate a grid board image ready for printing.
 *
 * @param img          output grayscale image (CV_8UC1) containing the full board
 * @param boardSize    board layout as columns × rows (e.g. `cv::Size(4, 3)`)
 * @param dict         dictionary used for the markers
 * @param markerSizePix  side length of each marker in pixels (default 50);
 *                     the output image will be `boardSize.width * markerSizePix` ×
 *                     `boardSize.height * markerSizePix` pixels
 * @param ids          optional custom marker id list in row-major order;
 *                     if empty, ids 0…(cols*rows−1) are used
 *
 * Markers are laid out in row-major order with no gap between them.
 * Pass the same `boardSize`, `dict`, and `ids` to detectBoard() for detection.
 */
CV_WRAP void generateBoardImage(OutputArray img, Size boardSize, DictionaryType dict,
                                int markerSizePix = 50, std::vector<int> ids = {});


/** @brief Detect a rectangular grid board of ArUco markers.
 *
 * @param image        input image (grayscale or BGR)
 * @param gridSize     board layout as columns × rows (e.g. `cv::Size(4, 3)` for a 4×3 grid)
 * @param dict         dictionary used to print the board
 * @param board        output Board populated with the detected markers
 * @param detectorParams  detection tuning parameters
 * @param ids          optional custom marker id list in row-major order;
 *                     if empty, ids 0…(cols*rows−1) are assumed
 * @return             true if at least one board marker was detected
 *
 * In Python the return value and output parameter are combined:
 * @code{.py}
 * found, board = cv.aruco2.detectBoard(image, (4, 3), cv.aruco2.DICT_6X6_250)
 * @endcode
 */
CV_WRAP bool detectBoard(InputArray image, cv::Size gridSize, DictionaryType dict,
                         CV_OUT Board &board, const DetectorParameters &detectorParams = {},
                         std::vector<int> ids = {});



/** @brief Compute image and object points for a detected board to pass to solvePnP().
 *
 * @param board      a detected board returned by detectBoard()
 * @param imgPoints  output array of marker corner image coordinates for all detected markers (CV_32FC2)
 * @param objPoints  output array of corresponding 3-D object points in board coordinates (CV_32FC3).
 *                   The board origin is at the top-left marker corner; X points right, Y points
 *                   down, Z=0.  Units are marker-side lengths; scale by the physical marker size
 *                   before calling solvePnP().
 *
 * Only detected markers are included, so `imgPoints` and `objPoints` are always the same length
 * even when the board is partially occluded.
 *
 * @sa getSolvePnpPoints(const Marker, OutputArray, OutputArray)
 */
CV_WRAP void getSolvePnpPoints(const Board board, OutputArray imgPoints, OutputArray objPoints);

/** @brief A detected ChArUco2-style diamond marker.
 *
 * A diamond is a 2×2 block of ArUco markers (standard on black squares, inverted on white).
 * Its identity is the combination of the four constituent marker ids, accessible via `id`
 * (as a `Vec4i` convenience field) or individually through each `markers[i].id`.
 *
 * @sa detectDiamonds, getSolvePnpPoints(const Diamond, OutputArray, OutputArray)
 */
struct CV_EXPORTS_W_SIMPLE Diamond {
    CV_PROP_RW cv::Vec4i id;                   ///< ids of the 4 constituent markers (clockwise from top-left)
    CV_PROP_RW std::vector<Marker> markers;    ///< the 4 detected markers forming the diamond
    CV_PROP_RW DictionaryType dict;            ///< dictionary used for the 4 markers
};


/** @brief Detect ChArUco2-style diamond markers in an image.
 *
 * A diamond is a 2×2 block of ArUco markers (standard on black squares, inverted on white).
 * Each detected Diamond carries the 4 constituent Marker objects and their combined id as a
 * `Vec4i` for convenient access.
 *
 * @param image        input image (grayscale or BGR)
 * @param dict         dictionary used to print the diamond markers
 * @param detectorParams  detection tuning parameters
 * @return             vector of detected Diamond objects; empty if none found
 */
CV_WRAP std::vector<Diamond> detectDiamonds(InputArray image, DictionaryType dict,
                                          const DetectorParameters &detectorParams = {});


/** @brief Compute image and object points for a detected diamond to pass to solvePnP().
 *
 * @param diamond    a detected diamond returned by detectDiamonds()
 * @param imgPoints  output array of marker corner image coordinates for all 4 markers (CV_32FC2)
 * @param objPoints  output array of corresponding 3-D object points in diamond coordinates
 *                   (CV_32FC3).  The origin is at the top-left marker corner; X points right,
 *                   Y points down, Z=0.  Units are marker-side lengths; scale by the physical
 *                   marker size before calling solvePnP().
 *
 * @sa getSolvePnpPoints(const Marker, OutputArray, OutputArray),
 *     getSolvePnpPoints(const Board, OutputArray, OutputArray)
 */
CV_WRAP void getSolvePnpPoints(const Diamond diamond, OutputArray imgPoints, OutputArray objPoints);

//! @}

}
