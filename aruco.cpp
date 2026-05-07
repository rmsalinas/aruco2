#include "aruco.hpp"
#include "aruco_dictionary.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>

namespace   {
using namespace cv;
using namespace cv :: aruco2;

/** @brief The MarkerDetector class is detecting the markers in the image passed */
class MarkerDetector{
public:
    // The only function you need to call
    static inline std::vector<Marker> detect(const cv::Mat &img, const std::vector<DictionaryType> dicts,  const DetectorParameters &params=DetectorParameters(),std::vector<Marker> *candidatesOut=nullptr);
private:
    static inline Marker sort( const  Marker &marker);
    static inline float  getSubpixelValue(const cv::Mat &im_grey,const cv::Point2f &p);
    static inline int   getMarkerId(  cv::Mat  candidateBits,int &idx, int &nrotations, const DetectorParameters &params,Dictionary &dict);
    static inline int isInto(const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) ;
    static std::vector<std::vector<cv::Point>> visitedAwareTracingContour(cv::Mat &padded, size_t minSize = 1,float maxRevisited=0.1) ;
    static int getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize) ;
    static void thres255Adaptive(cv::Mat &in,cv::Mat &out,int off=2,int thres=5);
};

namespace _private {
struct Homographer{
    Homographer(const std::vector<cv::Point2f> & out ){
        std::vector<cv::Point2f>  in={cv::Point2f(0,0),cv::Point2f(1,0),cv::Point2f(1,1),cv::Point2f(0,1)};
        H=cv::getPerspectiveTransform(in, out);
    }
    cv::Point2f operator()(const cv::Point2f &p){
        double *m=H.ptr<double>(0);
        double c=m[6]*p.x+m[7]*p.y+m[8];
        return cv::Point2f((m[0]*p.x+m[1]*p.y+m[2])/c,(m[3]*p.x+m[4]*p.y+m[5])/c);
    }
    cv::Mat H;
};
}
//Marker intersection. Tells the marker with most corners into another. 0 if no intersection or tie
int MarkerDetector::isInto(const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) {
    // Lambda for point-in-polygon test (Ray Casting)
    auto countInside = [](const std::vector<cv::Point2f>& source, const std::vector<cv::Point2f>& target) -> int {
        int count = 0;
        for (const auto& pt : source) {
            bool inside = false;
            // Fixed 4-side loop logic
            for (int i = 0, j = 3; i < 4; j = i++) {
                if (((target[i].y > pt.y) != (target[j].y > pt.y)) &&
                    (pt.x < (target[j].x - target[i].x) * (pt.y - target[i].y) / (target[j].y - target[i].y) + target[i].x)) {
                    inside = !inside;
                }
            }
            if (inside) count++;
        }
        return count;
    };
    // Count how many corners of A are in B
    int aInB = countInside(a, b);
    // Count how many corners of B are in A
    int bInA = countInside(b, a);
    // Rule 1: Must contain at least one corner
    if (aInB == 0 && bInA == 0) return 0;
    // Rule 2: Compare counts
    if (aInB > bInA) return 1;
    if (bInA > aInB) return 2;
    // Default: Tie or no relative dominance
    return 0;
}

std::vector<Marker>  MarkerDetector::detect(const cv::Mat &img,   const std::vector<DictionaryType> dicts,const DetectorParameters &params,std::vector<Marker> *candidatesOut){
    cv::Mat bwimage,thresImage;
    std::vector<Marker> DetectedMarkers;
    //first, convert to bw
    if(img.channels()==3)
        cv::cvtColor(img,bwimage,cv::COLOR_BGR2GRAY);
    else bwimage=img;
    /////////////////// Adaptive Threshold to detect border
    //    cv::adaptiveThreshold(bwimage, thresImage, 255.,cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, params.boxFilterSize, params.Thres);
    //this method is achieves a ~1.5 speed up
    cv::boxFilter( bwimage, thresImage, bwimage.type(), cv::Size(params.boxFilterSize, params.boxFilterSize),cv::Point(-1,-1), true, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED );
    thresImage=thresImage-bwimage;
    cv::threshold(thresImage, thresImage, params.thres, 255, cv::THRESH_BINARY);
    /////////////////// compute marker candidates by detecting contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> approxCurve;
    cv::RNG rand;
    //cv::findContours(thresImage, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    int  minSizeSq=params.minSize*params.minSize,minSize4=4*params.minSize;
    contours=visitedAwareTracingContour(thresImage,minSize4,params.maxTimesRevisited);

    //decide where to store the candidates. If candidatesOut is not null, store there, otherwise use a local variable
    std::vector<Marker> candidateslocal;
    if(candidatesOut!=nullptr) {
        candidatesOut->clear();
    }
    else{
        candidatesOut=&candidateslocal;
    }
    ///////////////// for each contour, approx to a rectangle
    for (unsigned int i = 0; i < contours.size(); i++)
    {
        // can approximate to a convex rect?
        cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.03, true);
        if (approxCurve.size() != 4 || !cv::isContourConvex(approxCurve)) continue;
        //check distance  between corners at least minSize pix
        if(  ((approxCurve[0].x-approxCurve[1].x)*(approxCurve[0].x-approxCurve[1].x) + (approxCurve[0].y-approxCurve[1].y)*(approxCurve[0].y-approxCurve[1].y))<minSizeSq) continue;
        if(  ((approxCurve[1].x-approxCurve[2].x)*(approxCurve[1].x-approxCurve[2].x) + (approxCurve[1].y-approxCurve[2].y)*(approxCurve[1].y-approxCurve[2].y))<minSizeSq) continue;
        if(  ((approxCurve[2].x-approxCurve[3].x)*(approxCurve[2].x-approxCurve[3].x) + (approxCurve[2].y-approxCurve[3].y)*(approxCurve[2].y-approxCurve[3].y))<minSizeSq) continue;
        if(  ((approxCurve[3].x-approxCurve[0].x)*(approxCurve[3].x-approxCurve[0].x) + (approxCurve[3].y-approxCurve[0].y)*(approxCurve[3].y-approxCurve[0].y))<minSizeSq) continue;
        // // add the points
        Marker marker;marker.corners.reserve(4);
        for (int j = 0; j < 4; j++)
            marker.corners.emplace_back( cv::Point2f( approxCurve[j].x,approxCurve[j].y));
        //sort corner in clockwise direction
        marker=sort(marker);
        candidatesOut->push_back(marker);
    }


    //now, for each candidate check bits inside
    for(size_t di=0;di<dicts.size();di++){
        Dictionary dict=getPredefinedDictionary(dicts[di]);
        std::vector<Marker> currDirMarkerDetected;
        cv::Mat bits(dict.markerSize+2,dict.markerSize+2,CV_8UC1),bitadaptive(dict.markerSize+2,dict.markerSize+2,CV_8UC1);

        for(auto it=candidatesOut->begin();it!=candidatesOut->end();){
            auto marker=*it;

            ////// extract the code. Obtain the intensities of the bits using  homography
            for(int i=0;i<int(params.maxAttemptsPerCandidate) && marker.id==-1;i++){
                //if not first attempt, we may wanna produce small random alteration of the corners
                auto marker2=marker;
                if( i!=0) for(int c=0;c<4;c++) {marker2.corners[c].x+=rand.gaussian(0.75);marker2.corners[c].y+=rand.gaussian(0.75);}//if not first, alter corner location
                _private::Homographer hom(marker2.corners);
                for(int r=0;r<bits.rows;r++){
                    for(int c=0;c<bits.cols;c++){
                        bits.at<uchar>(r,c)=uchar(0.5+getSubpixelValue(bwimage,hom(cv::Point2f(  float(c+0.5) / float(bits.cols) ,  float(r+0.5) / float(bits.rows)  ))));
                    }
                }
                if(i==2){ // if not working the first time, try this time adaptive threshold into the bits to improve robustness to lighting
                    thres255Adaptive(bits,bitadaptive);
                    bitadaptive.copyTo(bits);
                }
                else{
                    cv::threshold(bits,bits,0,255,cv::THRESH_OTSU);
                }
                //now, analyze the inner code to see it if is a marker. If so, rotate to have the points properly sorted
                int nrotations=0;
                if(getMarkerId(bits,marker.id,nrotations,params,dict)==0) continue;
                std::rotate(marker.corners.begin(),marker.corners.begin() + 4 - nrotations,marker.corners.end());
            }
            if(marker.id!=-1) {
                marker.dict= dicts[di];
                currDirMarkerDetected.push_back(marker);
                //remove from candidate list
                it=candidatesOut->erase(it);
            }
            else it++;//go to next
        }

        /// REMOVAL OF INNER DUPLICATED DETECTIONS OF THE SAME MARKER(INNER AND OUTER BORDER)
        std::sort(currDirMarkerDetected.begin(), currDirMarkerDetected.end(),[](const Marker &a,const Marker &b){return a.id<b.id;});
        {
            std::vector<bool> toRemove(currDirMarkerDetected.size(), false);
            for (int i = 0; i < int(currDirMarkerDetected.size()) - 1; i++)
            {
                for (int j = i + 1; j < int(currDirMarkerDetected.size()) && !toRemove[i]; j++)
                {
                    if (currDirMarkerDetected[i].id == currDirMarkerDetected[j].id )
                    {
                        auto res=isInto(currDirMarkerDetected[i].corners,currDirMarkerDetected[j].corners);
                        if( res==1)toRemove[i]=true;
                        else if( res==2)toRemove[j]=true;

                    }
                }
            }
            //now move to DetectedMarkers these not marked for removal
            for (unsigned int i = 0; i < currDirMarkerDetected.size(); i++)
                if (!toRemove[i]) DetectedMarkers.push_back(currDirMarkerDetected[i]);

        }
    }


    ////// finally subpixel corner refinement
    if(DetectedMarkers.size()>0){
        // TODO: cols/cols always equals 1 — halfwsize is always 4 regardless of image resolution
#warning "NEEDS TO BE CHANGED"
        int halfwsize= 4*float(bwimage.cols)/float(bwimage.cols) +0.5 ;
        std::vector<cv::Point2f> Corners;
        for (const auto &m:DetectedMarkers)
            Corners.insert(Corners.end(), m.corners.begin(),m.corners.end());
        cv::cornerSubPix(bwimage, Corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
        // copy back to the markers
        for (unsigned int i = 0; i < DetectedMarkers.size(); i++)
            for (int c = 0; c < 4; c++) DetectedMarkers[i].corners[c] = Corners[i * 4 + c];
    }
    return DetectedMarkers;//DONE
}
/**
 * @brief Tries to identify one candidate given the dictionary
 * @return candidate typ. zero if the candidate is not valid,
 *                           1 if the candidate is a black candidate (default candidate)
 *                           2 if the candidate is a white candidate
 */
int MarkerDetector:: getMarkerId(cv::Mat candidateBits, int &idx, int &nrotations, const DetectorParameters &params,Dictionary &dict){
    uint8_t typ=1;

    if(params.detectInvertedMarker ) candidateBits=~candidateBits;
    // analyze border bits
    int maximumErrorsInBorder =int(dict.markerSize * dict.markerSize * params.maxErroneousBitsInBorderRate);
    int borderErrors =getBorderErrors(candidateBits, dict.markerSize, params.markerBorderBits);
    if(borderErrors > maximumErrorsInBorder) return 0; // border is wrong
    // take only inner bits
    cv::Mat onlyBits =candidateBits.rowRange(params.markerBorderBits,candidateBits.rows - params.markerBorderBits).colRange(params.markerBorderBits, candidateBits.cols - params.markerBorderBits);
    onlyBits/=255;
    // try to indentify the marker
    if(!dict.identify(onlyBits, idx, nrotations, params.errorCorrectionRate))
        return 0;
    return typ;
}
/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
int MarkerDetector::getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize) {
    int sizeWithBorders = markerSize + 2 * borderSize;
    int totalErrors = 0;
    for(int y = 0; y < sizeWithBorders; y++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(y)[k] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(y)[sizeWithBorders - 1 - k] != 0) totalErrors++;
        }
    }
    for(int x = borderSize; x < sizeWithBorders - borderSize; x++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(k)[x] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(sizeWithBorders - 1 - k)[x] != 0) totalErrors++;
        }
    }
    return totalErrors;
}
float MarkerDetector::getSubpixelValue(const cv::Mat &im_grey, const cv::Point2f &p) {
    // 1. Get integer coordinates
    const int ix = static_cast<int>(p.x);
    const int iy = static_cast<int>(p.y);


    //   Boundary Check: Ensure the 2x2 patch is within limits
    // We check ix+1 and iy+1 because the interpolation looks at the next pixel over.
    if (ix < 0 || iy < 0 || ix >= im_grey.cols - 1 || iy >= im_grey.rows - 1) {
        // Option A: Return a default value
        // Option B: Clamp the point to the nearest valid boundary
        return 0.0f;
    }

    // 2. Get fractional parts
    const float dx = p.x - ix;
    const float dy = p.y - iy;
    // 3. Optimized Pointer Access
    const uchar* ptr = im_grey.ptr<uchar>(iy) + ix;
    const size_t step = im_grey.step;
    // 4. Fetch the four pixels immediately as floats
    const float p00 = static_cast<float>(ptr[0]);        // Top-Left
    const float p01 = static_cast<float>(ptr[1]);        // Top-Right
    const float p10 = static_cast<float>(ptr[step]);     // Bottom-Left
    const float p11 = static_cast<float>(ptr[step + 1]); // Bottom-Right
    // 5. Separable Interpolation (3 Multiplications total)
    const float top = p00 + dx * (p01 - p00);// Interpolate Top Row Horizontally
    const float bot = p10 + dx * (p11 - p10);    // Interpolate Bottom Row Horizontallys
    // Interpolate Vertically between Top and Bottom results
    return top + dy * (bot - top);
}
Marker  MarkerDetector::sort( const  Marker &marker){
    Marker res_marker=marker;
    /// sort the points in anti-clockwise order
    double dx1 = res_marker.corners[1].x - res_marker.corners[0].x;
    double dy1 = res_marker.corners[1].y - res_marker.corners[0].y;
    double dx2 = res_marker.corners[2].x - res_marker.corners[0].x;
    double dy2 = res_marker.corners[2].y - res_marker.corners[0].y;
    double o = (dx1 * dy2) - (dy1 * dx2);
    // if the third point is in the left side, then sort in anti-clockwise order
    if (o < 0.0)  std::swap(res_marker.corners[1], res_marker.corners[3]);
    return res_marker;
}

/**
 * @brief Traces the contours of a binary image using our visited aware Tracing algorithm.
 *
 * This function scans a binary image (foreground as 255, background as 0) and
 * finds the external boundaries of all distinct objects.
 */
std::vector<std::vector<cv::Point>> MarkerDetector::visitedAwareTracingContour(cv::Mat &padded, size_t minSize, float maxRevisited ) {
    if (padded.empty() || padded.type() != CV_8UC1) return {};
    // 1. Fast Initialization and Padding
    int rows = padded.rows;
    int cols = padded.cols;
    int32_t step = padded.step;
    uchar* data = padded.data;
    // Fast clear of top and bottom rows
    memset(data, 0, cols);
    memset(data + (rows - 1) * step, 0, cols);
    // Fast clear of left and right columns
    for (int r = 1; r < rows - 1; ++r) {
        uchar* row_ptr = data + r * step;
        row_ptr[0] = 0;
        row_ptr[cols - 1] = 0;
    }
    // 2. Precompute Neighbor Offsets based on image stride This removes the need for Point arithmetic in the loop
    const int offsets[16]={-1,-step-1,-step,-step+1,1,step+1,step,step-1, -1,-step-1,-step,-step+1,1,step+1,step,step-1, };
    // Use static tables to avoid initialization overhead on every call // 8-connectivity offsets relative to center (0,0) // Order: W, NW, N, NE, E, SE, S, SW
    const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 }, dy[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };
    // Pre-allocate results
    std::vector<std::vector<cv::Point>> contours;contours.reserve(2048);
    std::vector<cv::Point> buffer;buffer.reserve(2048);
    const uchar FOREGROUND = 255, BACKGROUND = 0,VISITED = 100;
    // 3. Scanning Loop
    // We iterate using raw pointers for maximum speed
    int rowStep=1;//std::max(1,int(minSize/6));
    for (int r = 1; r < rows - 1; r+=rowStep) {
        uchar* row_ptr = data + r * step;
        for (int c = 1; c < cols - 1;  ) {
            ////findStartContourPoint
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                cv::v_uint8 v_zero = cv::vx_setzero_u8();
                for (; c <= cols - cv::VTraits<cv::v_uint8>::vlanes(); c+= cv::VTraits<cv::v_uint8>::vlanes())
                {
                    cv::v_uint8 vmask = (cv::v_ne(cv::vx_load((uchar*)(row_ptr + c)), v_zero));
                    if (v_check_any(vmask))
                    {
                        c += v_scan_forward(vmask);
                        break;
                    }
                }
#endif
                //process last tail
                for (; c < cols && !row_ptr[c]; ++c) ;//last tail
            }
            if( c==cols) break;//reached end of row
            if (row_ptr[c] == FOREGROUND ) {// --- 4. Tracing Loop  if is foreground
                buffer.clear();
                int curr_x = c, curr_y = r,search_idx = 1 ;
                uchar* curr_ptr = row_ptr + c,*start_ptr=curr_ptr;
                size_t ntimesRevisited=0;
                do {
                    buffer.emplace_back(curr_x, curr_y);// Add point
                    *curr_ptr = VISITED;// Mark as visited
                    //showImage(padded);
                    // Search for next foreground pixel. We search 8 neighbors starting from search_idx
                    for (int i = 0; i < 8; ++i) {
                        int idx = search_idx + i; // index into offsets (0..15)
                        uchar* neighbor = curr_ptr + offsets[idx]; // Fast pointer arithmetic
                        if (*neighbor != BACKGROUND) {
                            // Found next boundary pixel
                            curr_ptr = neighbor;
                            int dir = (idx & 7);                             // Update Integer Coordinates using the small static tables(Use modulo 8 to get the distinct direction 0-7)
                            int next_x=curr_x+dx[dir], next_y=curr_y+dy[dir];
                            ntimesRevisited+= int(*neighbor == VISITED);
                            curr_x = next_x;curr_y = next_y;
                            search_idx = (dir + 5) & 7;
                            break;
                        }
                    }
                } while (curr_ptr != start_ptr );
                size_t bufsize=buffer.size();
                if (ntimesRevisited<= float(bufsize)*maxRevisited && bufsize >= minSize) {
                    contours.push_back(buffer);
                }
            }
            c++;//move to next pixel
            ////findEndContourPoint
            if ( row_ptr[c]){
                {
#if (CV_SIMD || CV_SIMD_SCALABLE)

                    cv::v_uint8 v_zero = cv::vx_setzero_u8();
                    for (; c <=  cols - cv::VTraits<cv::v_uint8>::vlanes(); c += cv::VTraits<cv::v_uint8>::vlanes())
                    {
                        cv::v_uint8 vmask = (cv::v_eq(cv::vx_load((uchar*)(row_ptr + c)), v_zero));
                        if (cv::v_check_any(vmask))
                        {
                            c += cv::v_scan_forward(vmask);
                            break;
                        }
                    }
#endif
                }
                for (; c < cols && row_ptr[c]; ++c) ;//last tail
            }
        }
    }
    return contours;
}

void MarkerDetector::thres255Adaptive(cv::Mat &in,cv::Mat &out,int off,int thres){
    cv::boxFilter( in, out, in.type(), cv::Size(off*2+1, off*2+1),
                  cv::Point(-1,-1), true, 4 );

    for(int i = 0; i < in.rows; i++ )
    {
        const uchar* sdata = in.ptr(i);
        uchar* ddata = out.ptr(i);
        for(int j = 0; j < in.cols; j++ )
            ddata[j] = ((ddata[j]-thres  )< sdata[j]) *255;
    }

}
}

namespace cv :: aruco2{

std::vector<Marker> detectMarkers(InputArray image,const std::vector<DictionaryType> &dicts,const DetectorParameters &detectorParams){
    return MarkerDetector::detect(image.getMat(),dicts,detectorParams);
}
std::vector<Marker> detectMarkers(InputArray image,DictionaryType dict,const DetectorParameters &detectorParams){
    return MarkerDetector::detect(image.getMat(),{dict},detectorParams);

}

}