/*
 * Nanofractal is a simplified version of the Aruco Fractal marker.
 *
 * With this you will be able to detect fractal markers easily. In addition, make use of the potential of fractal
 * markers, robust to occlusions and providing information from all corners of the marker (internal and external).
 *
 * The library detects the predefined fractal markers: https://drive.google.com/file/d/1JO3V-CQIScHu2U_wwKK7kcZ0qteYbSpu/view?usp=sharing
 *
 * You only need to define the marker you are going to use (FRACTAL_3L_6, FRACTAL_4L_6,...), to create the
 * MarkerDetector object. Then call the detect method with the input image as parameter. For example,
 * nanofractal::MarkerDetector("FRACTAL_5L_6"); (See Example1).
 *
 * If you besides the four corners of each detected marker, need all visible corners (also inners corners) of the marker
 * you should call the detect method with the image and the 2d/3d point vectors as parameters (See Example2).
 *
 * Note that the 3d points of the marker are normalized, if you need real 3d information you must indicate the
 * size of the marker when you create the detector. For example, nanofractal::MarkerDetector("FRACTAL_3L_6", 0.85);
 *
 *
 *
 * // Example1: Fractal marker detection
 *
 * auto image=cv::imread("image.jpg");
 * nanofractal::MarkerDetector TheDetector = nanofractal::MarkerDetector("FRACTAL_5L_6");
 * auto markers=TheDetector.detect(image);
 * for(const auto &m:markers)
 *    m.draw(image);
 * cv::imwrite("/path/to/out.png",image);
 *
 *
 * //Example2: Fractal marker detection and get 3d/2d correspondences

 * auto image=cv::imread("image.jpg");
 * nanofractal::MarkerDetector TheDetector = nanofractal::MarkerDetector("FRACTAL_5L_6", 0.85);
 * std::vector<cv::Point2f>p2d; std::vector<cv::Point3f>p3d;
 * auto markers=TheDetector.detect(image, p3d, p2d);
 * //Here you can call solvepnp using p3d and p2d points
 * for(auto pt:p2d)
 *    cv::circle(image,pt,5,cv::Scalar(0,0,255), cv::FILLED);
 * for(const auto &m:markers)
 *    m.draw(image);
 * cv::imwrite("/path/to/out.png",image);
 *
 *
 * If you use this file in your research, you must cite:
 *
 * 1. "Fractal Markers: A New Approach for Long-Range Marker Pose Estimation Under Occlusion,", F. J. Romero-Ramirez,
 * R. Muñoz-Salinas and R. Medina-Carnicer, in IEEE Access, vol. 7, pp. 169908-169919, year 2019.
 * 2. "Speeded up detection of squared fiducial markers", Francisco J. Romero-Ramirez, Rafael Muñoz-Salinas, Rafael
 * Medina-Carnicer, Image and Vision Computing, vol 76, pages 38-47, year 2018.
 *
 *  If you have any further question, please contact fj.romero[at]uco[dot]es
*/

#include "aruco2.hpp"
#ifndef _ARUCONanoFractal_H_
#define _ARUCONanoFractal_H_
#define FractalNanoVersion 4
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <map>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <cstring>
/**
 * The FractalMarkerDetector class detects fractal markers in the images passed
 *
namespace nanofractal{
 class FractalMarkerDetector{
  public:
    //@param fractal_config possible values (FRACTAL_2L_6,FRACTAL_3L_6,FRACTAL_4L_6,FRACTAL_5L_6)
    void setParams(std::string config, float markerSize=-1);
    inline std::vector<FractalMarker> detect(const cv::Mat &img);
    inline std::vector<FractalMarker> detect(const cv::Mat &img, std::vector<cv::Point3f>& p3d,
                                             std::vector<cv::Point2f>& p2d);
  };
}
*/


namespace nanofractal {
namespace _private{
namespace  picoflann {
struct L2{

    template<typename ElementType, typename ElementType2, typename Adapter>
    double compute_distance( const ElementType &elema,const ElementType2 &elemb,const Adapter & adapter,int ndims ,double worstDist)const
    {
        //compute dist
        double sqd=0;
        for(int i=0;i<ndims;i++) {
            double d= adapter(elema,i)-adapter(elemb,i);
            sqd+=d*d;
            if (sqd>worstDist) return sqd;
        }
        return sqd;
    }
};

template<int DIMS,typename Adapter,typename DistanceType=L2 >
class KdTreeIndex{

public:
    /**
     *Builds the index using the data  passes in your container and the adapter
     */
    template<typename Container  >
    inline void build(const Container &container ){
        _index.clear();
        _index.reserve(container.size()*2);
        _index.dims=DIMS;
        _index.nValues=container.size();
        //Create root and assign all items
        all_indices.resize(container.size());
        for(size_t i=0;i<container.size();i++)  all_indices[i]=i;
        if (container.size()==0) return;
        computeBoundingBox<Container>(_index.rootBBox,0,all_indices.size(),container);
        _index.push_back(Node());
        divideTree<Container>(_index,0,0,all_indices.size(),_index.rootBBox ,container);
    }


    inline void clear(){
        _index.clear();
        all_indices.clear();
    }

    //saves to a stream. Note that the container is not saved!
    inline void toStream (std::ostream &str)const;
    //reads from an stream. Note that the container is not readed!
    inline void fromStream (std::istream &str);

    template<  typename Type,typename Container >
    inline std::vector<std::pair<uint32_t,double> >  searchKnn(const Container &container,const Type &val,  int nn,bool sorted=true){
        std::vector<std::pair<uint32_t,double> > res;
        generalSearch<Type,Container>(res,container,val,-1,sorted,nn);
        return res;
    }


    template<  typename Type,typename Container >
    inline std::vector<std::pair<uint32_t,double> >  radiusSearch(const Container &container,const Type &val, double dist,bool sorted=true, int maxNN=-1)const{
        std::vector<std::pair<uint32_t,double> > res;
        generalSearch< Type,Container>(res,container,val,dist,sorted,maxNN);
        return res;
    }


    template< typename Type,typename Container >
    inline void  radiusSearch(std::vector<std::pair<uint32_t,double> > &res,const Container &container,const Type &val, double dist,bool sorted=true, int maxNN=-1){
        generalSearch<Type,Container>(res,container,val,dist,sorted,maxNN);
    }



private:

    struct Node{
        inline bool isLeaf()const{return _ileft==-1 && _iright==-1;}
        inline void setNodesInfo(uint32_t l,uint32_t r){_ileft=l; _iright=r;}
        double div_val;
        uint16_t col_index;//column index of the feature vector
        std::vector<int> idx;
        float divhigh,divlow;
        int64_t _ileft=-1,_iright=-1;//children
        void toStream(std::ostream &str) const;
        void fromStream(std::istream &str);
    };


    typedef std::vector<std::pair<double,double> > BoundingBox;

    struct Index:public  std::vector<Node>{
        BoundingBox rootBBox;
        int dims=0;
        int nValues=0;//number of elements of the set when call to build
        inline void toStream(std::ostream &str)const;
        inline void fromStream(std::istream &str);
    };
    Index _index;
    DistanceType _distance;
    Adapter adapter;
    //next are only used during build
    std::vector<uint32_t> all_indices;
    int _maxLeafSize=10 ;



    //temporal used during creation of the tree
    template< typename Container >
    void  divideTree(Index &index,uint64_t  nodeIdx,int startIndex,int endIndex ,BoundingBox &bbox,const Container&container){
        // std::cout<<"CREATE="<<startIndex<<"-"<<endIndex<<"|";toStream(std::cout,bbox);
        Node &currNode=index[nodeIdx];
        int count=endIndex-startIndex;
        assert(startIndex<endIndex);

        if (count<=  _maxLeafSize){
            currNode.idx.resize(count);
            for(int i=0;i<count;i++)
                currNode.idx[i]= all_indices[startIndex+i];
            computeBoundingBox<Container>(bbox,startIndex,endIndex,container);
            //  std::cout<<std::endl;
            return;
        }


        currNode.setNodesInfo( index.size(), index.size()+1);
        index.push_back(Node());
        int leftNode=index.size()-1;
        index.push_back(Node());
        int rightNode=index.size()-1;


        ///SELECT THE COL (DIMENSION) ON WHICH PARTITION IS MADE
        if (0){
            BoundingBox _bbox;
            computeBoundingBox<Container>(_bbox,startIndex,endIndex,container);
            //        //get the dimension with highest distnaces
            double max_spread=-1;
            currNode.col_index=0;
            for(int i=0;i<DIMS;i++){
                double spread=_bbox[i].second-_bbox[i].first;// maxV[i]-minV[i];
                if ( spread>max_spread){
                    max_spread=spread;
                    currNode.col_index=i;
                }
            }
            //select the split val
            double split_val= (bbox[currNode.col_index].first + bbox[currNode.col_index].second) / 2;
            if (split_val < _bbox[currNode.col_index].first) currNode.div_val = _bbox[currNode.col_index].first;
            else if (split_val > _bbox[currNode.col_index].second  ) currNode.div_val = _bbox[currNode.col_index].second ;
            else  currNode.div_val = split_val;
        }
        else{
            ///SELECT THE COL (DIMENSION) ON WHICH PARTITION IS MADE
            double var[DIMS],mean[DIMS];
            //compute the variance of the features to  select the highest one
            mean_var_calculate<Container>(startIndex,endIndex, var, mean,container);
            currNode.col_index=0;
            //select element with highest variance
            for(int i=1;i<DIMS;i++)
                if (var[i]>var[currNode.col_index]) currNode.col_index=i;

            //now sort all indices according to the selected value

            currNode.div_val=mean[currNode.col_index];
        }





        //compute the variance of the features to  select the highest one
        //now sort all indices according to the selected value

        //std::cout<<" CUT FEAT="<<currNode.col_index<< " VAL="<<currNode.div_val<<std::endl;
        int lim1,lim2;
        planeSplit<Container> ( &all_indices[startIndex],count,currNode.col_index,currNode.div_val,lim1,lim2,container);

        int split_index;

        if (lim1>count/2) split_index = lim1;
        else if (lim2<count/2) split_index = lim2;
        else split_index = count/2;

        //        /* If either list is empty, it means that all remaining features
        //              * are identical. Split in the middle to maintain a balanced tree.
        //              */
        if ((lim1==count)||(lim2==0)) split_index = count/2;
        //create partitions with at least minLeafSize elements
        if (_maxLeafSize!=1)
            if ( split_index<_maxLeafSize || count-split_index<_maxLeafSize) {
                std::sort(all_indices.begin()+ startIndex ,all_indices.begin()+endIndex,[&](const uint32_t &a,const uint32_t&b){
                    return adapter(container.at(a), currNode.col_index)<adapter(container.at(b),currNode.col_index);
                });
                split_index=count/2;
                currNode.div_val=adapter(container.at(all_indices[startIndex+split_index]),currNode.col_index);
            }



        //  currNode.div_val=_features.ptr<float>(all_indices[split_index])[currNode.col_index];

        BoundingBox left_bbox(bbox);
        left_bbox[currNode.col_index].second = currNode.div_val;
        divideTree<Container>( index,leftNode ,startIndex,startIndex+split_index,left_bbox,container);
        left_bbox[currNode.col_index].second = currNode.div_val;
        assert(left_bbox[currNode.col_index].second <=currNode.div_val);
        BoundingBox right_bbox(bbox);
        right_bbox[currNode.col_index].first = currNode.div_val;
        divideTree<Container>(index,rightNode,startIndex+split_index,endIndex,right_bbox,container);

        currNode.divlow = left_bbox[currNode.col_index].second;
        currNode.divhigh = right_bbox[currNode.col_index].first;
        assert(currNode.divlow<=currNode.divhigh);

        for (int i=0; i<DIMS; ++i) {
            bbox[i].first = std::min(left_bbox[i].first, right_bbox[i].first);
            bbox[i].second = std::max(left_bbox[i].second, right_bbox[i].second);
        }
    }




    template<  typename Container >
    void  computeBoundingBox (BoundingBox& bbox, int start,int end,const Container&container ){
        bbox.resize(DIMS);
        for (int i=0; i<DIMS; ++i)
            bbox[i].second = bbox[i].first = adapter( container.at(  all_indices[start]),i);

        for (int k=start+1; k<end; ++k) {
            for (int i=0; i<DIMS; ++i) {
                float v=    adapter( container.at(all_indices[k]),i);
                if (v<bbox[i].first) bbox[i].first = v;
                if (v>bbox[i].second) bbox[i].second = v;
            }
        }
    }

    template<  typename Container >
    void  mean_var_calculate(  int startindex, int endIndex, double var[], double mean[],const Container&container){
        const int MAX_ELEM_MEAN=100;
        //recompute centers
        //compute new center
        memset(mean,0,sizeof(double)*DIMS);
        double sum2[DIMS];
        memset(sum2,0,sizeof(double)*DIMS);
        //finish when at least MAX_ELEM_MEAN elements computed
        int cnt=0;
        //std::min(MAX_ELEM_MEAN,endIndex-startindex );
        int increment=1;
        if ( endIndex-startindex>=2*MAX_ELEM_MEAN) increment=(endIndex-startindex)/MAX_ELEM_MEAN;
        for(int i=startindex;i<endIndex;i+=increment) {
            for(int c=0;c<DIMS;c++)    {
                auto val= adapter(container.at(all_indices[i]),c);
                mean[c] += val;
                sum2[c] += val*val;
            }
            cnt++;
        }

        double invcnt=1./double(cnt);
        for(int c=0;c<DIMS;c++) {
            mean[c]*=invcnt;
            var[c]= sum2[c]*invcnt - mean[c]*mean[c];
        }
    }


    /**
     *  Subdivide the list of points by a plane perpendicular on axe corresponding
     *  to the 'cutfeat' dimension at 'cutval' position.
     *
     *  On return:
     *  dataset[ind[0..lim1-1]][cutfeat]<cutval
     *  dataset[ind[lim1..lim2-1]][cutfeat]==cutval
     *  dataset[ind[lim2..count]][cutfeat]>cutval
     */
    template<  typename Container >
    void  planeSplit(uint32_t* ind, int count, int cutfeat, float cutval, int& lim1, int& lim2,const Container&container){
        /* Move vector indices for left subtree to front of list. */
        int left = 0;
        int right = count-1;
        for (;; ) {
            while (left<=right && adapter(container.at( ind[left]),cutfeat)<cutval) ++left;
            while (left<=right &&   adapter(container.at( ind[right]),cutfeat)>=cutval) --right;
            if (left>right) break;
            std::swap(ind[left], ind[right]); ++left; --right;
        }
        lim1 = left;
        right = count-1;
        for (;; ) {
            while (left<=right &&   adapter(container.at(ind[left]),cutfeat)<=cutval) ++left;
            while (left<=right &&   adapter(container.at(ind[right]),cutfeat)>cutval) --right;
            if (left>right) break;
            std::swap(ind[left], ind[right]); ++left; --right;
        }
        lim2 = left;
    }


    template< typename Type >
    inline  double computeInitialDistances(const Type &elem, double dists[ ],const BoundingBox &bbox) const{
        float distsq = 0.0;

        for (int i = 0; i <DIMS; ++i) {
            double elem_i=adapter( elem,i);
            if (elem_i < bbox[i].first) {
                auto d=elem_i-bbox[i].first;
                dists[i] = d*d;// distance_.accum_dist(vec[i], root_bbox_[i].first, i);
                distsq += dists[i];
            }
            if (elem_i > bbox[i].second) {
                auto d=elem_i-bbox[i].second;
                dists[i] = d*d;//distance_.accum_dist(vec[i], root_bbox_[i].second, i);
                distsq += dists[i];
            }
        }
        return distsq;
    }
    //THe function that does the search in all exact methods
    template< typename Type,typename Container>
    inline void generalSearch( std::vector<std::pair<uint32_t,double> > &res, const Container&container,const Type &val,double dist,bool sorted=true,uint32_t maxNn=std::numeric_limits<int>::max() )const{
        double  dists[DIMS];
        memset(dists ,0,sizeof(double)*DIMS);
        res.clear();
        ResultSet hres( res ,maxNn,dist>0?dist*dist:-1.f);
        float distsq = computeInitialDistances<Type>(val, dists,_index.rootBBox);
        searchExactLevel<Type,Container> (_index,0,val,hres,distsq,dists,1,container);
        if (sorted && res.size()>1)
            std::sort(res.begin(),res.end(),[](const std::pair<uint32_t,double>&a,const std::pair<uint32_t,double>&b){return a.second<b.second;});
    }

    //heap having at the top the maximum element

    class ResultSet{
    public:
        std::vector<std::pair<uint32_t,double> > &array;
        int maxSize;
        double maxValue=std::numeric_limits<double>::max();
        bool radius_search=false;

    public:
        ResultSet(  std::vector<std::pair<uint32_t,double> > &data_ref,uint32_t MaxSize=std::numeric_limits<uint32_t>::max(),double MaxV=-1):array(data_ref){
            maxSize=MaxSize;
            //set value for radius search
            if (MaxV>0){
                maxValue =MaxV;
                radius_search=true;
            }
        }


        inline void push (const  std::pair<uint32_t,double> &val)
        {
            if ( radius_search &&  val.second<maxValue){
                array.push_back(val);
            }
            else{
                if (array.size()>=size_t(maxSize) ) {
                    //check if the maxium must be replaced by this
                    if ( val.second<array[0].second){
                        swap(array.front() ,array.back());
                        array.pop_back();
                        if (array.size()> 1)  up (0) ;
                    }
                    else return;
                }
                array.push_back(val);
                if (array.size()>1) down ( array.size()-1) ;
            }
            //            array_size++;
        }

        inline double worstDist()const{
            if (radius_search)return maxValue;//radius search
            else if (array.size()<size_t(maxSize))return std::numeric_limits<double>::max();
            return array[0].second;
        }
        inline double top()const{assert(!array.empty()); return array[0].second;}
    private:
        inline void  down ( size_t index)
        {
            if(index==0) return;
            size_t parentIndex =(index - 1) / 2;
            if (array[parentIndex].second< array[ index].second ) {
                swap( array[index],array[parentIndex] );
                down (parentIndex) ;
            }
        }
        inline void up (size_t index)
        {
            size_t leftIndex  = 2 * index + 1 ;//vl_heap_left_child (index) ;
            size_t rightIndex = 2 * index + 2;//vl_heap_right_child (index) ;

            /* no childer: stop */
            if (leftIndex >= array.size()) return ;

            /* only left childer: easy */
            if (rightIndex >= array.size()) {
                if ( array [ index].second <array[leftIndex].second)
                    swap (  array [ index], array[leftIndex]) ;
                return ;
            }

            /* both childern */
            {
                if ( array[ rightIndex].second< array[  leftIndex].second  ) {
                    /* swap with left */
                    if (array [index].second< array[leftIndex].second ) {
                        swap ( array [index] ,  array[leftIndex]) ;
                        up ( leftIndex) ;
                    }
                } else {
                    /* swap with right */
                    if ( array[ index].second  < array[rightIndex].second) {
                        swap ( array[ index], array[rightIndex]) ;
                        up ( rightIndex) ;
                    }
                }
            }
        }
    };

    template< typename Type,typename Container >
    inline  void searchExactLevel(const Index &index,int64_t nodeIdx,const Type &elem, ResultSet  &res, double mindistsq, double  dists[ ],double epsError ,const Container &container)const{

        const Node &currNode=index[nodeIdx];
        if (currNode.isLeaf()){
            double worstDist=res.worstDist();
            for(size_t i=0;i<currNode.idx.size();i++){
                double sqd=_distance.compute_distance(elem,container.at(currNode.idx[i]),adapter,DIMS,worstDist);
                if (sqd<worstDist) {
                    res.push( {currNode.idx[i],sqd});
                    worstDist=res.worstDist();
                }
            }
        }
        else{

            double val = adapter( elem, currNode.col_index);
            double diff1 = val - currNode.divlow;
            double diff2 = val - currNode.divhigh;

            uint32_t bestChild;
            uint32_t otherChild;
            double cut_dist;
            if ((diff1+diff2)<0) {
                bestChild = currNode._ileft;
                otherChild = currNode._iright;
                cut_dist = diff2*diff2 ;
            }
            else {
                bestChild =  currNode._iright;
                otherChild = currNode._ileft;
                cut_dist =  diff1*diff1;
            }
            /* Call recursively to search next level down. */
            searchExactLevel<Type,Container> (index,bestChild,elem,res, mindistsq, dists ,epsError,container );

            float dst = dists[currNode.col_index];
            mindistsq = mindistsq + cut_dist - dst;
            dists[currNode.col_index] = cut_dist;
            if (mindistsq*epsError <=res.worstDist())
                searchExactLevel<Type,Container>   (index,otherChild,elem,res, mindistsq, dists,epsError,container );
            dists[currNode.col_index] = dst;
        }
    }

};
template<int DIMS,typename AAdapter,typename DistanceType>
void KdTreeIndex<DIMS,AAdapter,DistanceType>::Node::toStream(std::ostream &str) const{
    str.write((char*)&div_val,sizeof(div_val));
    str.write((char*)&col_index,sizeof(col_index));
    str.write((char*)&divhigh,sizeof(divhigh));
    str.write((char*)&divlow,sizeof(divlow));
    str.write((char*)&_ileft,sizeof(_ileft));
    str.write((char*)&_iright,sizeof(_iright));
    uint64_t s=idx.size();
    str.write((char*)&s,sizeof(s));
    str.write((char*)&idx[0],sizeof(idx[0])*idx.size());

}

template<int DIMS,typename AAdapter,typename DistanceType>
void KdTreeIndex<DIMS,AAdapter,DistanceType>::Node::fromStream(std::istream &str){
    str.read((char*)&div_val,sizeof(div_val));
    str.read((char*)&col_index,sizeof(col_index));
    str.read((char*)&divhigh,sizeof(divhigh));
    str.read((char*)&divlow,sizeof(divlow));
    str.read((char*)&_ileft,sizeof(_ileft));
    str.read((char*)&_iright,sizeof(_iright));
    uint64_t s;
    str.read((char*)&s,sizeof(s));
    idx.resize(s);
    str.read((char*)&idx[0],sizeof(idx[0])*idx.size());

}

template<int DIMS,typename AAdapter,typename DistanceType>
void KdTreeIndex<DIMS,AAdapter,DistanceType>::Index::toStream(std::ostream &str)const
{

    str.write((char*)&dims,sizeof(dims));
    str.write((char*)&rootBBox[0],sizeof(rootBBox[0])*dims);
    str.write((char*)&nValues,sizeof(nValues));

    uint64_t s=std::vector<Node>::size();
    str.write((char*)&s,sizeof(s));
    for(size_t i=0;i<std::vector<Node>::size();i++) std::vector<Node>::at(i).toStream(str);
}

template<int DIMS,typename AAdapter,typename DistanceType>
void KdTreeIndex<DIMS,AAdapter,DistanceType>::Index::fromStream(std::istream &str){
    str.read((char*)&dims,sizeof(dims));
    rootBBox.resize(dims);
    str.read((char*)&rootBBox[0],sizeof(rootBBox[0])*dims);
    str.read((char*)&nValues,sizeof(nValues));


    uint64_t s;;
    str.read((char*)&s,sizeof(s));
    std::vector<Node>::resize(s);
    for(size_t i=0;i<std::vector<Node>::size();i++) std::vector<Node>::at(i).fromStream(str);
    if (dims!=DIMS && this->size()!=0 && nValues!=0)
        throw std::runtime_error("Number of dimensions of the index in the stream is different from the number of dimensions of this");

}

template<int DIMS,typename AAdapter,typename DistanceType>
void KdTreeIndex<DIMS,AAdapter,DistanceType>::toStream (std::ostream &str)const{
    _index.toStream(str);
}

template<int DIMS,typename AAdapter,typename DistanceType>
void KdTreeIndex<DIMS,AAdapter,DistanceType>::fromStream(std::istream &str){
    _index.fromStream(str);
}
}
struct Homographer{
    Homographer(const std::vector<cv::Point2f> & out ){
        std::vector<cv::Point2f>  in={cv::Point2f(0,0),cv::Point2f(1,0),cv::Point2f(1,1),cv::Point2f(0,1)};
        H=cv::getPerspectiveTransform(in, out);
    }
    cv::Point2f operator()(const cv::Point2f &p){
        double *m=H.ptr<double>(0);
        double a=m[0]*p.x+m[1]*p.y+m[2];
        double b=m[3]*p.x+m[4]*p.y+m[5];
        double c=m[6]*p.x+m[7]*p.y+m[8];
        return cv::Point2f(a/c,b/c);
    }
    cv::Mat H;
};
struct PicoFlann_KeyPointAdapter{
    inline  float operator( )(const cv::KeyPoint &elem, int dim)const { return dim==0?elem.pt.x:elem.pt.y; }
    inline  float operator( )(const cv::Point2f &elem, int dim)const { return dim==0?elem.x:elem.y; }
};

/* KeyPoints Filter. Delete kpoints with low response and duplicated. */
void kfilter(std::vector<cv::KeyPoint> &kpoints)
{
    float minResp = kpoints[0].response;
    float maxResp = kpoints[0].response;
    for (auto &p:kpoints){
        p.size=40;
        if(p.response < minResp) minResp = p.response;
        if(p.response > maxResp) maxResp = p.response;
    }
    float thresoldResp = (maxResp - minResp) * 0.20f + minResp;

    for(uint32_t xi=0; xi<kpoints.size();xi++)
    {
        //Erase keypoints with low response (20%)
        if(kpoints[xi].response < thresoldResp){
            kpoints[xi].size=-1;
            continue;
        }

        //Duplicated keypoints (closer)
        for(uint32_t xj=xi+1; xj<kpoints.size();xj++)
        {
            if(pow(kpoints[xi].pt.x - kpoints[xj].pt.x,2) + pow(kpoints[xi].pt.y - kpoints[xj].pt.y,2) < 100)
            {
                if(kpoints[xj].response > kpoints[xi].response)
                    kpoints[xi] = kpoints[xj];

                kpoints[xj].size=-1;
            }
        }
    }
    kpoints.erase(std::remove_if(kpoints.begin(),kpoints.end(), [](const cv::KeyPoint &kpt){return kpt.size==-1;}), kpoints.end());
}

/*Corners classification*/
void assignClass(const cv::Mat &im, std::vector<cv::KeyPoint>& kpoints, float sizeNorm=0.f, int wsize=5)
{
    if(im.type()!=CV_8UC1)
        throw std::runtime_error("assignClass Input image must be 8UC1");
    int wsizeFull=wsize*2+1;

    cv::Mat labels = cv::Mat::zeros(wsize*2+1,wsize*2+1,CV_8UC1);
    cv::Mat thresIm=cv::Mat(wsize*2+1,wsize*2+1,CV_8UC1);

    for(auto &kp:kpoints)
    {
        float x = kp.pt.x;
        float y = kp.pt.y;

        //Convert point range from norm (-size/2, size/2) to (0,imageSize)
        if(sizeNorm>0){
            x = im.cols * (x/sizeNorm + 0.5f);
            y = im.rows * (-y/sizeNorm + 0.5f);
        }

        x= int(x+0.5f);
        y= int(y+0.5f);

        cv::Rect r= cv::Rect(x-wsize,y-wsize,wsize*2+1,wsize*2+1);
        //Check boundaries
        if(r.x<0 || r.x+r.width>im.cols || r.y<0 ||
            r.y+r.height>im.rows) continue;

        int endX=r.x+r.width;
        int endY=r.y+r.height;
        uchar minV=255,maxV=0;
        for(int y=r.y; y<endY; y++){
            const uchar *ptr=im.ptr<uchar>(y);
            for(int x=r.x; x<endX; x++)
            {
                if(minV>ptr[x]) minV=ptr[x];
                if(maxV<ptr[x]) maxV=ptr[x];
            }
        }

        if ((maxV-minV) < 25) {
            kp.class_id=0;
            continue;
        }

        double thres=(maxV+minV)/2.0;

        unsigned int nZ=0;
        //count non zero considering the threshold
        for(int y=0; y<wsizeFull; y++){
            const uchar *ptr=im.ptr<uchar>( r.y+y)+r.x;
            uchar *thresPtr= thresIm.ptr<uchar>(y);
            for(int x=0; x<wsizeFull; x++){
                if( ptr[x]>thres) {
                    nZ++;
                    thresPtr[x]=255;
                }
                else thresPtr[x]=0;
            }
        }
        //set all to zero labels.setTo(cv::Scalar::all(0));
        for(int y=0; y<thresIm.rows; y++){
            uchar *labelsPtr=labels.ptr<uchar>(y);
            for(int x=0; x<thresIm.cols; x++) labelsPtr[x]=0;
        }

        uchar newLab = 1;
        std::map<uchar, uchar> unions;
        for(int y=0; y<thresIm.rows; y++){
            uchar *thresPtr=thresIm.ptr<uchar>(y);
            uchar *labelsPtr=labels.ptr<uchar>(y);
            for(int x=0; x<thresIm.cols; x++)
            {
                uchar reg = thresPtr[x];
                uchar lleft_px = 0;
                uchar ltop_px = 0;

                if(x-1>-1 && reg==thresPtr[x-1])
                    lleft_px =labelsPtr[x-1];

                if(y-1>-1 && reg==thresIm.ptr<uchar>(y-1)[x])
                    ltop_px =  labels.at<uchar>(y-1, x);

                if(lleft_px==0 && ltop_px==0)
                    labelsPtr[x] = newLab++;

                else if(lleft_px!=0 && ltop_px!=0)
                {
                    if(lleft_px < ltop_px)
                    {
                        labelsPtr[x]  = lleft_px;
                        unions[ltop_px] = lleft_px;
                    }
                    else if(lleft_px > ltop_px)
                    {
                        labelsPtr[x]  = ltop_px;
                        unions[lleft_px] = ltop_px;
                    }
                    //Same
                    else labelsPtr[x]  = ltop_px;
                }
                else
                    if(lleft_px!=0) labelsPtr[x]  = lleft_px;
                    else labelsPtr[x]  = ltop_px;
            }
        }

        int nc= newLab-1 - unions.size();
        if(nc==2)
            if(nZ > thresIm.total()-nZ) kp.class_id = 0;
            else kp.class_id = 1;
        else if (nc > 2)
            kp.class_id = 2;
    }
}
}

/**
 * @brief The Markers that belong to the fractal marker
 */
class FractalMarker : public std::vector<cv::Point2f>
{
public:
    FractalMarker(int id, cv::Mat m, std::vector<cv::Point3f> corners, std::vector<int> id_submarkers);
    FractalMarker(){};

    inline int nBits() { return _M.total(); }
    inline cv::Mat mat(){ return _M; }
    inline cv::Mat mask(){ return _mask; }
    inline std::vector<int> subMarkers(){ return _submarkers; }
    void addSubFractalMarker(FractalMarker submarker);
    // returns the distance of the marker side
    inline float getMarkerSize() const
    {
        return static_cast<float>(cv::norm(keypts[0].pt  - keypts[1].pt));
    }
    inline std::vector<cv::KeyPoint> getKeypts();
    inline void draw(cv::Mat &image,const cv::Scalar color=cv::Scalar(0,0,255))const;

    int id;
    std::vector<cv::KeyPoint> keypts; //Corners & class. First 4 corners are external
private:
    cv::Mat _M;
    cv::Mat _mask;
    std::vector<int> _submarkers;
};


FractalMarker::FractalMarker(int id, cv::Mat m, std::vector<cv::Point3f> corners, std::vector<int> id_submarkers)
{
    this->id = id;
    this->_M = m;
    for(auto pt:corners)
        keypts.push_back(cv::KeyPoint(pt.x,pt.y,-1,-1,-1,-1,0));
    _submarkers = id_submarkers;
    _mask = cv::Mat::ones(m.size(), CV_8UC1);
}

std::vector<cv::KeyPoint> FractalMarker::getKeypts()
{
    if(keypts.size() > 4) return keypts;

    int nBitsSquared = int(sqrt(mat().total()));
    float bitSize =  getMarkerSize() / (nBitsSquared+2);

    //Set submarker pixels (=1) and add border
    cv::Mat marker;
    mat().copyTo(marker);
    marker +=  -1 * (mask()-1);
    cv::Mat markerBorder;
    copyMakeBorder(marker, markerBorder, 1,1,1,1,cv::BORDER_CONSTANT,0);

    //Get inner corners
    for(int y=0; y< markerBorder.rows-1; y++)
    {
        for(int x=0; x< markerBorder.cols-1; x++)
        {
            int sum = markerBorder.at<uchar>(y, x) + markerBorder.at<uchar>(y, x+1) +
                      markerBorder.at<uchar>(y+1, x) + markerBorder.at<uchar>(y+1, x+1);

            if(sum==1)
                keypts.push_back(cv::KeyPoint(cv::Point2f(x-nBitsSquared/2.f,-(y-nBitsSquared/2.f))*bitSize,-1,-1,-1,-1,1));
            else if(sum==3)
                keypts.push_back(cv::KeyPoint(cv::Point2f(x-nBitsSquared/2.f,-(y-nBitsSquared/2.f))*bitSize,-1,-1,-1,-1,0));
            else if(sum==2)
            {
                if((markerBorder.at<uchar>(y, x) == markerBorder.at<uchar>(y+1, x+1)) && (markerBorder.at<uchar>(y, x+1) == markerBorder.at<uchar>(y+1, x)))
                    keypts.push_back(cv::KeyPoint(cv::Point2f(x-nBitsSquared/2.f,-(y-nBitsSquared/2.f))*bitSize,-1,-1,-1,-1,2));
            }
        }
    }

    return keypts;
}

void FractalMarker::addSubFractalMarker(FractalMarker submarker)
{
    int nBitsSqrt= sqrt(nBits());
    float bitSize = getMarkerSize() / (nBitsSqrt+2.0f);
    float nsubBits = submarker.getMarkerSize() / bitSize;

    int x_min = int(round(submarker.keypts[0].pt.x / bitSize + nBitsSqrt/2));
    int x_max = x_min + nsubBits;
    int y_min = int(round(-submarker.keypts[0].pt.y / bitSize + nBitsSqrt/2));
    int y_max = y_min + nsubBits;

    for(int y=y_min; y<y_max; y++){
        for(int x=x_min; x<x_max; x++){
            _mask.at<uchar>(y,x)=0;
        }
    }
}

void FractalMarker::draw(cv::Mat &in, const cv::Scalar color) const{
    float flineWidth=  std::max(1.f, std::min(5.f, float(in.cols) / 500.f));
    int lineWidth= round( flineWidth);
    for(int i=0;i<4;i++)
        cv::line(in, (*this)[i], (*this)[(i+1 )%4], color, lineWidth);

    auto p2 =  cv::Point2f(2.f * static_cast<float>(lineWidth), 2.f * static_cast<float>(lineWidth));
    cv::rectangle(in, (*this)[0] - p2, (*this)[0] + p2, cv::Scalar(0, 0, 255, 255), -1);
    cv::rectangle(in, (*this)[1] - p2, (*this)[1] + p2, cv::Scalar(0, 255, 0, 255), lineWidth);
    cv::rectangle(in, (*this)[2] - p2, (*this)[2] + p2, cv::Scalar(255, 0, 0, 255), lineWidth);
}

/**
 * @brief The FractalMarkerSet configurations
 */
class FractalMarkerSet
{
public:
    FractalMarkerSet(){};
    FractalMarkerSet(std::string config);
    void convertToMeters(float size);

    //Fractal configuration. id_marker
    std::map<int, FractalMarker> fractalMarkerCollection;
    //Correspondence number of bits and marker ids
    std::map<int, std::vector<int>> bits_ids;
    // variable indicates if the data is expressed in meters or in pixels or are normalized
    int mInfoType;/* -1:NONE, 0:PIX, 1:METERS, 2:NORMALIZE*/
    int idExternal;
};

void FractalMarkerSet::convertToMeters(float size)
{
    if (!(mInfoType == 0 || mInfoType == 2))
        throw std::runtime_error("The FractalMarkers are not expressed in pixels or normalized");

    mInfoType = 1;

    // now, get the size of a pixel, and change scale
    float pixSizeM = size / float(fractalMarkerCollection[idExternal].getMarkerSize());

    for (size_t i=0; i < fractalMarkerCollection.size(); i++)
        for(auto &kpt:fractalMarkerCollection[i].keypts)
            kpt.pt *= pixSizeM;
}

FractalMarkerSet::FractalMarkerSet(std::string str)
{
    std::stringstream stream;
    if (str=="FRACTAL_2L_6")
    {
        unsigned char _conf_2L_6[] = {
            0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x24, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0xbe, 0xab, 0xaa, 0xaa, 0x3e,
            0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0x3e, 0xab, 0xaa, 0xaa, 0x3e,
            0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0x3e, 0xab, 0xaa, 0xaa, 0xbe,
            0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0xbe, 0xab, 0xaa, 0xaa, 0xbe,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00
        };
        unsigned int _conf_2L_6_len = 272;
        stream.write((char*) _conf_2L_6, sizeof(unsigned char)*_conf_2L_6_len);
    }
    else if (str=="FRACTAL_3L_6")
    {
        unsigned char _conf_3L_6[] = {
            0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01,
            0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
            0xb7, 0x6d, 0xdb, 0xbe, 0xb7, 0x6d, 0xdb, 0x3e, 0x00, 0x00, 0x00, 0x00,
            0xb7, 0x6d, 0xdb, 0x3e, 0xb7, 0x6d, 0xdb, 0x3e, 0x00, 0x00, 0x00, 0x00,
            0xb7, 0x6d, 0xdb, 0x3e, 0xb7, 0x6d, 0xdb, 0xbe, 0x00, 0x00, 0x00, 0x00,
            0xb7, 0x6d, 0xdb, 0xbe, 0xb7, 0x6d, 0xdb, 0xbe, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01,
            0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0xbe,
            0x25, 0x49, 0x12, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0x3e,
            0x25, 0x49, 0x12, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0x3e,
            0x25, 0x49, 0x12, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0xbe,
            0x25, 0x49, 0x12, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        };
        unsigned int _conf_3L_6_len = 480;
        stream.write((char*) _conf_3L_6, sizeof(unsigned char)*_conf_3L_6_len);
    }
    else if (str=="FRACTAL_4L_6")
    {
        unsigned char _conf_4L_6[] = {
            0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0xa9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00,
            0x00, 0xef, 0xee, 0xee, 0xbe, 0xef, 0xee, 0xee, 0x3e, 0x00, 0x00, 0x00,
            0x00, 0xef, 0xee, 0xee, 0x3e, 0xef, 0xee, 0xee, 0x3e, 0x00, 0x00, 0x00,
            0x00, 0xef, 0xee, 0xee, 0x3e, 0xef, 0xee, 0xee, 0xbe, 0x00, 0x00, 0x00,
            0x00, 0xef, 0xee, 0xee, 0xbe, 0xef, 0xee, 0xee, 0xbe, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x01, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
            0x00, 0x64, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0xbe, 0xcd, 0xcc, 0x4c,
            0x3e, 0x00, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0x3e, 0xcd, 0xcc, 0x4c,
            0x3e, 0x00, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0x3e, 0xcd, 0xcc, 0x4c,
            0xbe, 0x00, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0xbe, 0xcd, 0xcc, 0x4c,
            0xbe, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00,
            0x00, 0x89, 0x88, 0x88, 0xbd, 0x89, 0x88, 0x88, 0x3d, 0x00, 0x00, 0x00,
            0x00, 0x89, 0x88, 0x88, 0x3d, 0x89, 0x88, 0x88, 0x3d, 0x00, 0x00, 0x00,
            0x00, 0x89, 0x88, 0x88, 0x3d, 0x89, 0x88, 0x88, 0xbd, 0x00, 0x00, 0x00,
            0x00, 0x89, 0x88, 0x88, 0xbd, 0x89, 0x88, 0x88, 0xbd, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00
        };
        unsigned int _conf_4L_6_len = 713;
        stream.write((char*) _conf_4L_6, sizeof(unsigned char)*_conf_4L_6_len);
    }
    else if (str=="FRACTAL_5L_6")
    {
        unsigned char _conf_5L_6[] = {
            0x02, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xa9, 0x00, 0x00,
            0x00, 0x4f, 0xec, 0xc4, 0xbe, 0x4f, 0xec, 0xc4, 0x3e, 0x00, 0x00, 0x00,
            0x00, 0x4f, 0xec, 0xc4, 0x3e, 0x4f, 0xec, 0xc4, 0x3e, 0x00, 0x00, 0x00,
            0x00, 0x4f, 0xec, 0xc4, 0x3e, 0x4f, 0xec, 0xc4, 0xbe, 0x00, 0x00, 0x00,
            0x00, 0x4f, 0xec, 0xc4, 0xbe, 0x4f, 0xec, 0xc4, 0xbe, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01,
            0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00,
            0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0xbe, 0x7d, 0xcb,
            0x37, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0x3e, 0x7d, 0xcb,
            0x37, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0x3e, 0x7d, 0xcb,
            0x37, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0xbe, 0x7d, 0xcb,
            0x37, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0xd9, 0x89,
            0x9d, 0xbd, 0xd9, 0x89, 0x9d, 0x3d, 0x00, 0x00, 0x00, 0x00, 0xd9, 0x89,
            0x9d, 0x3d, 0xd9, 0x89, 0x9d, 0x3d, 0x00, 0x00, 0x00, 0x00, 0xd9, 0x89,
            0x9d, 0x3d, 0xd9, 0x89, 0x9d, 0xbd, 0x00, 0x00, 0x00, 0x00, 0xd9, 0x89,
            0x9d, 0xbd, 0xd9, 0x89, 0x9d, 0xbd, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00,
            0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0xbc, 0x21, 0x0d,
            0xd2, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0x3c, 0x21, 0x0d,
            0xd2, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0x3c, 0x21, 0x0d,
            0xd2, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0xbc, 0x21, 0x0d,
            0xd2, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01,
            0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        };
        unsigned int _conf_5L_6_len = 898;
        stream.write((char*) _conf_5L_6, sizeof(unsigned char)*_conf_5L_6_len);
    }
    else
        throw std::runtime_error("Configuration no valid: "+str+". Use: FRACTAL_2L_6, FRACTAL_3L_6, FRACTAL_4L_6 or FRACTAL_5L_6.");

    stream.read((char*)&mInfoType,sizeof(mInfoType));
    /*Number of markers*/
    int _nmarkers;
    stream.read((char*)&_nmarkers,sizeof(_nmarkers));
    stream.read((char*)&idExternal,sizeof(idExternal));

    for(int i=0; i<_nmarkers; i++)
    {
        //ID
        int id;
        stream.read((char*)&id,sizeof(id));

        //NBITS
        int nbits;
        stream.read((char*)&nbits,sizeof(nbits));

        //CORNERS
        std::vector<cv::Point3f> corners(4);
        stream.read((char*)&corners[0],sizeof(cv::Point3f)*4);

        //MAT
        cv::Mat mat;
        mat.create(sqrt(nbits), sqrt(nbits), CV_8UC1);
        stream.read((char*)mat.data, mat.elemSize() * mat.total());

        //SUBMARKERS
        int nsub;
        stream.read((char*)&nsub,sizeof(nsub));
        std::vector<int> id_submarkers(nsub);
        if (nsub > 0)
            stream.read((char*)&id_submarkers[0],sizeof(int)*nsub);

        fractalMarkerCollection[id] = FractalMarker(id, mat, corners, id_submarkers);
    }

    //Add subfractals
    for(auto &id_marker:fractalMarkerCollection)
    {
        FractalMarker &marker = id_marker.second;
        for(auto id:id_marker.second.subMarkers())
            marker.addSubFractalMarker(fractalMarkerCollection[id]);

        //Init marker kpts
        marker.getKeypts();

        bits_ids[marker.nBits()].push_back(marker.id);
    }
}


/**
 * @brief The MarkerDetector class is detecting the markers in the image passed
 *
 */
class FractalMarkerDetector{
public:
    /**@param fractal_config possible values (FRACTAL_2L_6,FRACTAL_3L_6,FRACTAL_4L_6,FRACTAL_5L_6)
     */
    void setParams(std::string fractal_config, float markerSize=-1);
    inline std::vector<FractalMarker> detect(const cv::Mat &img);
    inline std::vector<FractalMarker> detect(const cv::Mat &img, std::vector<cv::Point3f>& p3d,
                                             std::vector<cv::Point2f>& p2d);
private:
    FractalMarkerSet fractalMarkerSet;
    static inline  std::vector<cv::Point2f> sort( const  std::vector<cv::Point2f> &marker);
    static inline  float  getSubpixelValue(const cv::Mat &im_grey,const cv::Point2f &p);
    static inline  int    getMarkerId(const cv::Mat &bits,int &nrotations, const std::vector<int>& markersId, const FractalMarkerSet& markerSet);
    static inline  int    perimeter(const std::vector<cv::Point2f>& a);

};


void FractalMarkerDetector::setParams(std::string config, float markerSize)
{
    fractalMarkerSet = FractalMarkerSet(config);
    if(markerSize != -1) fractalMarkerSet.convertToMeters(markerSize);

}

std::vector<FractalMarker> FractalMarkerDetector::detect(const cv::Mat &img, std::vector<cv::Point3f>& p3d,
                                                         std::vector<cv::Point2f>& p2d)
{
    cv::Mat bwimage;
    if(img.channels()==3)
        cv::cvtColor(img,bwimage,cv::COLOR_BGR2GRAY);
    else bwimage=img;

    //Fractal marker detection
    std::vector<FractalMarker> detected =  detect(bwimage);

    if(detected.size() > 0)
    {
        //External corners to compute homography
        std::vector<cv::Point2f>imgpoints;
        std::vector<cv::Point3f>objpoints;
        for(auto marker:detected)
        {
            for(auto p2d:marker)
                imgpoints.push_back(p2d);

            for(int c=0; c<4; c++)
            {
                cv::KeyPoint kpt = fractalMarkerSet.fractalMarkerCollection[marker.id].getKeypts()[c];
                objpoints.push_back(cv::Point3f(kpt.pt.x, kpt.pt.y, 0));
            }
        }

        //FAST
        std::vector<cv::KeyPoint> kpoints;
        cv::Ptr<cv::FastFeatureDetector> fd = cv::FastFeatureDetector::create();
        fd->detect(bwimage, kpoints);
        //Filter kpoints (low response) and removing duplicated.
        _private::kfilter(kpoints);
        _private::assignClass(bwimage, kpoints);

        _private::picoflann::KdTreeIndex<2,_private::PicoFlann_KeyPointAdapter>  kdtree;
        kdtree.build(kpoints);

        cv::Mat H = cv::findHomography(objpoints, imgpoints);

        for(auto &fm:fractalMarkerSet.fractalMarkerCollection)
        {
            std::vector<cv::Point2f> imgPoints;
            std::vector<cv::Point2f> objPoints;
            std::vector<cv::KeyPoint> objKeyPoints = fm.second.getKeypts();

            for(auto kpt : objKeyPoints)
                objPoints.push_back(cv::Point2f(kpt.pt.x, kpt.pt.y));

            cv::perspectiveTransform(objPoints, imgPoints, H);

            //We consider only markers whose internal points are separated by a specific distance.
            bool consider=true;
            for(size_t i=0; i<imgPoints.size()-1 && consider; i++)
                for(size_t j=i+1; j<imgPoints.size() && consider; j++)
                    if(pow(imgPoints[i].x-imgPoints[j].x, 2) + pow(imgPoints[i].y-imgPoints[j].y, 2) < 150)
                        consider=false;

            if(consider)
            {
                for(size_t idx=0; idx<imgPoints.size(); idx++)
                {
                    if(imgPoints[idx].x > 0 && imgPoints[idx].x < img.cols
                        && imgPoints[idx].y>0 && imgPoints[idx].y<img.rows)
                    {
                        std::vector<std::pair<uint32_t, double>> res = kdtree.radiusSearch(kpoints, imgPoints[idx], 10);
                        if(res.size() == 1)
                        {
                            if(kpoints[res[0].first].class_id == objKeyPoints[idx].class_id)
                            {
                                p2d.push_back(kpoints[res[0].first].pt);
                                p3d.push_back(cv::Point3f(objPoints[idx].x, objPoints[idx].y, 0));
                            }
                        }
                    }
                }
            }
            else
            {
                //If a marker is detected and it is not possible take all their corners,
                //at least take the external one!
                for(auto markerDetected:detected)
                {
                    if(markerDetected.id == fm.first)
                    {
                        for(int c=0; c<4; c++)
                        {
                            cv::Point2f pt = markerDetected.keypts[c].pt;
                            p3d.push_back(cv::Point3f(pt.x,pt.y,0));
                            p2d.push_back(markerDetected[c]);
                        }
                        break;
                    }
                }
            }
        }

        if(p2d.size()>0)
        {
            //corner subpixel
            cv::Size winSize = cv::Size(4, 4);
            cv::Size zeroZone = cv::Size( -1, -1 );
            cv::TermCriteria criteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005);      //cv::cornerSubPix(image, points, winSize, zeroZone, criteria);
            cornerSubPix(bwimage, p2d, winSize, zeroZone, criteria);
        }
    }
    return detected;
}

std::vector<FractalMarker>  FractalMarkerDetector::detect(const cv::Mat &img){

    cv::Mat bwimage,thresImage;

    std::vector<std::pair<int, std::vector<cv::Point2f>>> candidates;

    std::vector<FractalMarker> DetectedFractalMarkers;

    //first, convert to bw
    if(img.channels()==3)
        cv::cvtColor(img,bwimage,cv::COLOR_BGR2GRAY);
    else bwimage=img;


    ///////////////////////////////////////////////////
    // Adaptive Threshold to detect border
    int adaptiveWindowSize=std::max(int(3),int(15*float(bwimage.cols)/1920.));
    if( adaptiveWindowSize%2==0) adaptiveWindowSize++;
    cv::adaptiveThreshold(bwimage, thresImage, 255.,cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, adaptiveWindowSize, 7);

    ///////////////////////////////////////////////////
    // compute marker candidates by detecting contours
    //if image is eroded, minSize must be adapted
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> approxCurve;
    cv::findContours(thresImage, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    //analyze  it is a paralelepiped likely to be the marker
    for (unsigned int i = 0; i < contours.size(); i++)
    {
        // check it is a possible element by first checking that is is large enough
        if (120 > int(contours[i].size())  ) continue;
        // can approximate to a convex rect?
        cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.05, true);

        if (approxCurve.size() != 4 || !cv::isContourConvex(approxCurve)) continue;
        // add the points
        std::vector<cv::Point2f> markerCandidate;
        for (int j = 0; j < 4; j++)
            markerCandidate.push_back( cv::Point2f( approxCurve[j].x,approxCurve[j].y));

        //sort corner in clockwise direction
        markerCandidate=sort(markerCandidate);

        //extract the code
        //obtain the intensities of the bits using homography
        _private::Homographer hom(markerCandidate);

        for(auto b_vm:fractalMarkerSet.bits_ids)
        {
            int nbitsWithBorder = sqrt(b_vm.first)+2;
            cv::Mat bits(nbitsWithBorder,nbitsWithBorder,CV_8UC1);
            int pixelSum=0;

            for(int r=0;r<bits.rows;r++){
                for(int c=0;c<bits.cols;c++){
                    auto pixelValue=uchar(0.5+getSubpixelValue(bwimage,hom(cv::Point2f(  float(c+0.5) / float(bits.cols) ,  float(r+0.5) / float(bits.rows)  ))));
                    bits.at<uchar>(r,c)=pixelValue;
                    pixelSum+=pixelValue;
                }
            }

            //threshold by the average value
            double mean=double(pixelSum)/double(bits.cols*bits.rows);
            cv::threshold(bits,bits,mean,255,cv::THRESH_BINARY);

            //now, analyze the inner code to see if is a marker.
            //  If so, rotate to have the points properly sorted
            int nrotations=0;

            int id=getMarkerId(bits, nrotations, b_vm.second, fractalMarkerSet);

            if(id==-1) continue;//not a marker
            std::rotate(markerCandidate.begin(),markerCandidate.begin() + 4 - nrotations,markerCandidate.end());
            candidates.push_back(std::make_pair(id,markerCandidate));
        }
    }

    ////////////////////////////////////////////
    //remove duplicates
    // sort by id and within same id set the largest first
    std::sort(candidates.begin(), candidates.end(),[](const std::pair<int, std::vector<cv::Point2f>> &a,const std::pair<int, std::vector<cv::Point2f>> &b){
        if( a.first<b.first) return true;
        else if( a.first==b.first) return perimeter(a.second)>perimeter(b.second);
        else return false;
    });

    // Using std::unique remove duplicates
    auto ip = std::unique(candidates.begin(), candidates.end(),[](const std::pair<int, std::vector<cv::Point2f>> &a,const std::pair<int, std::vector<cv::Point2f>> &b){return a.first==b.first;});
    candidates.resize(std::distance(candidates.begin(), ip));

    if(candidates.size()>0){
        ////////////////////////////////////////////
        //finally subpixel corner refinement
        int halfwsize= 4*float(bwimage.cols)/float(bwimage.cols) +0.5 ;
        std::vector<cv::Point2f> Corners;
        for (const auto &m:candidates)
            Corners.insert(Corners.end(), m.second.begin(),m.second.end());
        cv::cornerSubPix(bwimage, Corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
        // copy back to the markers
        for (unsigned int i = 0; i < candidates.size(); i++)
        {
            DetectedFractalMarkers.push_back(fractalMarkerSet.fractalMarkerCollection[candidates[i].first]);
            for (int c = 0; c < 4; c++) DetectedFractalMarkers[i].push_back(Corners[i * 4 + c]);
        }
    }

    //Done
    return DetectedFractalMarkers;
}

int  FractalMarkerDetector::perimeter(const std::vector<cv::Point2f>& a)
{
    int sum = 0;
    for (size_t i = 0; i < a.size(); i++)
        sum+=cv::norm( a[i]-a[(i + 1) % a.size()]);
    return sum;
}

int FractalMarkerDetector:: getMarkerId(const cv::Mat &bits, int &nrotations, const std::vector<int>& markersId, const FractalMarkerSet& fmset){

    auto rotate=[](const cv::Mat& in)
    {
        cv::Mat out(in.size(),in.type());
        for (int i = 0; i < in.rows; i++)
            for (int j = 0; j < in.cols; j++)
                out.at<uchar>(i, j) = in.at<uchar>(in.cols - j - 1, i);
        return out;
    };

    //first check that outer is all black
    for(int x=0;x<bits.cols;x++){
        if( bits.at<uchar>(0,x)!=0)return -1;
        if( bits.at<uchar>(bits.rows-1,x)!=0)return -1;
        if( bits.at<uchar>(x,0)!=0)return -1;
        if( bits.at<uchar>(x,bits.cols-1)!=0)return -1;
    }

    //now, get the inner bits wo the black border
    cv::Mat bit_inner(bits.cols-2,bits.rows-2,CV_8UC1);
    for(int r=0;r<bit_inner.rows;r++)
        for(int c=0;c<bit_inner.cols;c++)
            bit_inner.at<uchar>(r,c)=bits.at<uchar>(r+1,c+1);

    nrotations = 0;
    do
    {
        for(auto idx:markersId)
        {
            FractalMarker fm = fmset.fractalMarkerCollection.at(idx);

            //Apply mask to substract submarkers

            cv::Mat masked;
            bit_inner.copyTo(masked, fm.mask());

            //Code without submarkers == fractal marker?
            if (cv::countNonZero(masked != fm.mat()*255) == 0)
                return idx;
        }
        bit_inner = rotate(bit_inner);
        nrotations++;
    } while (nrotations < 4);

    return -1;
}

float FractalMarkerDetector::getSubpixelValue(const cv::Mat &im_grey,const cv::Point2f &p){

    float intpartX;
    float decpartX=std::modf(p.x,&intpartX);
    float intpartY;
    float decpartY=std::modf(p.y,&intpartY);

    cv::Point tl;

    if (decpartX>0.5) {
        if (decpartY>0.5) tl=cv::Point(intpartX,intpartY);
        else tl=cv::Point(intpartX,intpartY-1);
    }
    else{
        if (decpartY>0.5) tl=cv::Point(intpartX-1,intpartY);
        else tl=cv::Point(intpartX-1,intpartY-1);
    }
    if(tl.x<0) tl.x=0;
    if(tl.y<0) tl.y=0;
    if(tl.x>=im_grey.cols)tl.x=im_grey.cols-1;
    if(tl.y>=im_grey.cols)tl.y=im_grey.rows-1;
    return (1.f-decpartY)*(1.-decpartX)*float(im_grey.at<uchar>(tl.y,tl.x))+
           decpartX*(1-decpartY)*float(im_grey.at<uchar>(tl.y,tl.x+1))+
           (1-decpartX)*decpartY*float(im_grey.at<uchar>(tl.y+1,tl.x))+
           decpartX*decpartY*float(im_grey.at<uchar>(tl.y+1,tl.x+1));
}


std::vector<cv::Point2f>  FractalMarkerDetector::sort( const  std::vector<cv::Point2f> &marker){
    std::vector<cv::Point2f>  res_marker=marker;
    /// sort the points in anti-clockwise order
    // trace a line between the first and second point.
    // if the thrid point is at the right side, then the points are anti-clockwise
    double dx1 = res_marker[1].x - res_marker[0].x;
    double dy1 = res_marker[1].y - res_marker[0].y;
    double dx2 = res_marker[2].x - res_marker[0].x;
    double dy2 = res_marker[2].y - res_marker[0].y;
    double o = (dx1 * dy2) - (dy1 * dx2);

    if (o < 0.0)
    {  // if the third point is in the left side, then sort in anti-clockwise order
        std::swap(res_marker[1], res_marker[3]);
    }
    return res_marker;
}
}
#endif

// ---------------------------------------------------------------------------
// cv::aruco2 wrappers
// ---------------------------------------------------------------------------
namespace cv {
namespace aruco2 {

static std::string fractalTypeName(FractalType ft) {
    switch (ft) {
        case FRACTAL_2L_6: return "FRACTAL_2L_6";
        case FRACTAL_3L_6: return "FRACTAL_3L_6";
        case FRACTAL_4L_6: return "FRACTAL_4L_6";
        case FRACTAL_5L_6: return "FRACTAL_5L_6";
    }
    return "FRACTAL_3L_6";
}

void generateFractalImage(OutputArray _img, const FractalType &ftype, int bitSize) {
    nanofractal::FractalMarkerSet fmset(fractalTypeName(ftype));

    auto &outer = fmset.fractalMarkerCollection.at(fmset.idExternal);
    int nBits = int(std::round(std::sqrt(float(outer.mat().total()))));
    int imgSize = (nBits + 2) * bitSize;

    cv::Mat img(imgSize, imgSize, CV_8UC1, cv::Scalar(255));

    // Render outer markers first, inner last so inner paints over outer
    std::vector<int> sortedIds;
    for (const auto &kv : fmset.fractalMarkerCollection)
        sortedIds.push_back(kv.first);
    std::sort(sortedIds.begin(), sortedIds.end(), [&](int a, int b) {
        return fmset.fractalMarkerCollection.at(a).getMarkerSize() >
               fmset.fractalMarkerCollection.at(b).getMarkerSize();
    });

    // Normalized coords [-1,+1] → pixel coords in img
    auto normToPix = [&](cv::Point2f n) -> cv::Point2f {
        float px = float(bitSize) + (n.x + 1.0f) * float(nBits * bitSize) / 2.0f;
        float py = float(bitSize) + (1.0f - n.y) * float(nBits * bitSize) / 2.0f;
        return {px, py};
    };

    for (int id : sortedIds) {
        auto &fm = fmset.fractalMarkerCollection.at(id);
        int mBits = int(std::round(std::sqrt(float(fm.mat().total()))));

        // Source: corners of the mBits×mBits bit matrix (col,row order = x,y)
        std::vector<cv::Point2f> srcPts = {
            {0.f, 0.f}, {float(mBits), 0.f},
            {float(mBits), float(mBits)}, {0.f, float(mBits)}
        };
        // Destination: fm.keypts[0..3] mapped to pixels
        // keypts ordering: TL(-1,+1), TR(+1,+1), BR(+1,-1), BL(-1,-1)
        std::vector<cv::Point2f> dstPts;
        for (int i = 0; i < 4; i++)
            dstPts.push_back(normToPix(fm.keypts[i].pt));

        cv::Mat markerBits;
        fm.mat().convertTo(markerBits, CV_8UC1, 255.0); // 0→0 (black), 1→255 (white)

        cv::Mat M = cv::getPerspectiveTransform(srcPts, dstPts);
        cv::warpPerspective(markerBits, img, M, img.size(),
                            cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
    }

    _img.assign(img);
}

std::vector<FractalMarker> detectFractals(InputArray _img, FractalType ftype) {
    cv::Mat img = _img.getMat();

    nanofractal::FractalMarkerDetector detector;
    detector.setParams(fractalTypeName(ftype));

    std::vector<cv::Point3f> p3d;
    std::vector<cv::Point2f> p2d;
    auto detected = detector.detect(img, p3d, p2d);

    // Need outer marker's normalized corner positions for the multi-marker fallback
    nanofractal::FractalMarkerSet fmset(fractalTypeName(ftype));
    auto &tmplOuter = fmset.fractalMarkerCollection.at(fmset.idExternal);

    std::vector<FractalMarker> result;
    for (size_t i = 0; i < detected.size(); i++) {
        FractalMarker m;
        m.type = ftype;
        m.id = detected[i].id;
        for (int c = 0; c < 4; c++)
            m.corners.push_back(detected[i][c]);

        if (detected.size() == 1 && !p2d.empty()) {
            // Single marker: use all correspondences (inner + outer corners)
            m.imgPoints = p2d;
            m.objPoints = p3d;
        } else {
            // Multiple markers: use the 4 outer corners per instance
            for (int c = 0; c < 4; c++) {
                m.imgPoints.push_back(detected[i][c]);
                m.objPoints.push_back(cv::Point3f(
                    tmplOuter.keypts[c].pt.x, tmplOuter.keypts[c].pt.y, 0.f));
            }
        }
        result.push_back(m);
    }
    return result;
}

void drawDetectedFractals(InputOutputArray _image, const std::vector<FractalMarker> &fractals,
                          Scalar color) {
    cv::Mat image = _image.getMat();

    float lineWidthF = std::max(1.f, std::min(5.f, float(image.cols) / 500.f));
    int lineWidth = int(std::round(lineWidthF));

    for (const auto &m : fractals) {
        if (m.corners.size() < 4) continue;

        for (int i = 0; i < 4; i++)
            cv::line(image, m.corners[i], m.corners[(i + 1) % 4], color, lineWidth);

        // Red dot on corners[0] to show orientation
        auto p2 = cv::Point2f(2.f * lineWidthF, 2.f * lineWidthF);
        cv::rectangle(image, m.corners[0] - p2, m.corners[0] + p2,
                      cv::Scalar(0, 0, 255), -1);

        // ID at centroid
        cv::Point2f center(0, 0);
        for (const auto &c : m.corners) center += c;
        center *= 0.25f;
        cv::putText(image, std::to_string(m.id), center,
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, lineWidth);
    }
}

void getSolvePnpPoints(const FractalMarker &fractal, OutputArray imgPoints,
                       OutputArray objPoints, float markerSize) {
    // 3D coords are in normalized space [-1, +1] (total span = 2 units)
    float scale = markerSize / 2.0f;
    int n = int(fractal.imgPoints.size());

    if (n == 0) {
        imgPoints.assign(cv::Mat());
        objPoints.assign(cv::Mat());
        return;
    }

    cv::Mat imgMat(n, 1, CV_32FC2);
    cv::Mat objMat(n, 1, CV_32FC3);
    for (int i = 0; i < n; i++) {
        imgMat.at<cv::Vec2f>(i) = {fractal.imgPoints[i].x, fractal.imgPoints[i].y};
        objMat.at<cv::Vec3f>(i) = {fractal.objPoints[i].x * scale,
                                   fractal.objPoints[i].y * scale,
                                   fractal.objPoints[i].z * scale};
    }
    imgPoints.assign(imgMat);
    objPoints.assign(objMat);
}

} // namespace aruco2
} // namespace cv

