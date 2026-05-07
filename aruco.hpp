// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#pragma once
#include <opencv2/core.hpp>
#include "aruco_dictionary.hpp"
#include <optional>
namespace cv :: aruco2 {

/** @brief struct DetectorParameters is used by detectMarkers
 */
struct CV_EXPORTS_W_SIMPLE DetectorParameters {
    int boxFilterSize=15,thres=3; //values for adaptive thresholding
    int minSize=10;//minimum size of a contour side to be considered as a marker candidate
    int maxAttemptsPerCandidate=5;//number of attempts to identify a candidate by slightly altering the corners
    // [0,1] ; maximum number of times a contour can revisit any of its pixels (1 is the minimum which is the starting point)
    //if you set a high value (std::numeric_limits<int>::max()) the algorithm behaves as the normal moore contour tracer
    float maxTimesRevisited=0.05; //1 equals tradional algo,
    /// number of bits of the marker border, i.e. marker border width (default 1).
    float  markerBorderBits=1; //i do not see this useful. all dicts have 1 border bit but its used in opencv  aruco and I keep it here
    double errorCorrectionRate=0;//The default 0.6 value in aruco opencv is very dangerous. It causes many false positives.
    double maxErroneousBitsInBorderRate=0;//maximum rate of erroneous bits in the border. Default 0 means no error allowed.
    bool detectInvertedMarker=false;//if the markers are printed in white over black background
};
/**
 * @brief A fiducial marker
 * It is a vector where each corner is a corner of the detected marker
 */
class CV_EXPORTS_W Marker : public std::vector<cv::Point2f>
{
public:
    // id of  the marker
    int id=-1;
    //id of the dict
    DictionaryType dict=DictionaryType(-1);
};

/** @brief Generate a canonical marker image
 *
 * @param dictionary dictionary of markers indicating the type of markers
 * @param id identifier of the marker that will be returned. It has to be a valid id in the specified dictionary.
 * @param sidePixels size of the image in pixels
 * @param img output image with the marker
 * @param borderBits width of the marker border.
 *
 * This function returns a marker image in its canonical form (i.e. ready to be printed)
 */
CV_EXPORTS_W void generateImageMarker(const DictionaryType &dictionary, int id, int sidePixels, OutputArray img,
                                      int borderBits = 1);



/** @brief Basic marker detection
     *
     * @param image input image
     * @param detectorParams marker detection parameters
     *
     * Performs marker detection in the input image.
     *
     * Note that this function does not perform pose estimation.
     * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort
     * input image with corresponding camera model, if camera parameters are known
     * @sa undistort
     */
CV_WRAP std::vector<Marker> detectMarkers(InputArray image,DictionaryType dict=DICT_ARUCO_MIP_36h12,const DetectorParameters &detectorParams={});
/**
 * @brief detectMarkers
 * @param image
 * @param dicts
 * @param detectorParams
 * @return
 */
CV_WRAP std::vector<Marker> detectMarkers(InputArray image,const std::vector<DictionaryType> &dicts,const DetectorParameters &detectorParams={});


/** @brief Draw detected markers in image
 *
 * @param image input/output image. It must have 1 or 3 channels. The number of channels is not altered.
 * @param markers detected on input image.
 * @param borderColor color of marker borders. Rest of colors (text color and first corner color)
 * are calculated based on this one to improve visualization.
 *
 * Given an array of detected markers, this functions draws them in the image.
 * The marker borders and identifiers are drawn.
 * Useful for debugging purposes.
 */
CV_EXPORTS_W void drawDetectedMarkers(InputOutputArray image, const std::vector<Marker> &markers, Scalar borderColor = Scalar(0, 255, 0));


/** @brief Board of ArUco markers
 *
 * A board is a set of markers in the 3D space with a common coordinate system.
 * The common form of a board of marker is a planar (2D) board, however any 3D layout can be used.
 * A Board object is composed by:
 * - The object points of the marker corners, i.e. their coordinates respect to the board system.
 * - The dictionary which indicates the type of markers of the board
 * - The identifier of all the markers in the board.
 */
class CV_EXPORTS_W Board{

        cv::Size bsize;
        DictionaryType dict;
        std::vector<int> ids;

        std::vector<cv::Point2f> corners;
        std::vector<Marker> markers;
};

/**
 * @brief detectBoard
 * @param image
 * @param bsize
 * @param dict
 * @param detectorParams
 * @param ids
 * @return
 */
CV_WRAP std::optional<Board> detectBoard(InputArray image,cv::Size bsize, DictionaryType dict, const DetectorParameters &detectorParams={},std::vector<int> ids={});
/**
 * @brief detectDiamons
 * @param image
 * @param dict
 * @param detectorParams
 * @return
 */
CV_WRAP std::vector<Board> detectDiamons(InputArray image, DictionaryType dict, const DetectorParameters &detectorParams={});


}
