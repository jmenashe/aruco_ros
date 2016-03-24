/**

Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.

*/
#include <aruco/arucofidmarkers.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
namespace aruco {

  /************************************
 *
 *
 *
 *
 ************************************/
  /**
*/
  Mat FiducialMarkers::createMarkerImage(int id,int size,int gsize) throw (cv::Exception)
  {
    Mat marker(size,size, CV_8UC1, Scalar(0));
    Mat m = getMarkerMat(id, gsize);
    for(int y = 0; y < m.rows; y++)
      for(int x = 0; x < m.cols; x++) {
        int swidth = size / (gsize + 2);
        Mat roi = marker(Rect((x+1)*swidth,(y+1)*swidth,swidth,swidth));
        if(m.at<uchar>(y,x) == 1) roi.setTo(255);
      }
    return marker;
  }
  /**
 *
 */
  cv::Mat FiducialMarkers::getMarkerMat(int id,int gsize) throw (cv::Exception)
  {
    Mat marker(gsize,gsize, CV_8UC1, Scalar(0));
    if(gsize == 5) {
      if (0<=id && id<1024) {
        //for each line, create
        int ids[4]={0x10,0x17,0x09,0x0e};
        for (int y=0;y<gsize;y++) {
          int index=(id>>2*(gsize-1-y)) & 0x0003;
          int val=ids[index];
          for (int x=0;x<gsize;x++) {
            if ( ( val>>(gsize-1-x) ) & 0x0001 ) marker.at<uchar>(y,x)=1;
            else marker.at<uchar>(y,x)=0;
          }
        }
      } else throw cv::Exception (9189,"Invalid marker id","aruco::fiducidal::getMarkerMat",__FILE__,__LINE__);
    } else {
      if(id < (1 << gsize * gsize)) {
        for(int y = 0; y < gsize; y++) {
          for(int x = 0; x < gsize; x++) {
            int pos = y * gsize + x;
            int val = (1 << pos) & id;
            val = val ? 1 : 0;
            marker.at<uchar>(y,x) = val;
          }
        }
      } else throw cv::Exception(9189,"Invalid marker id","aruco::fiducial::getMarkerMat",__FILE__,__LINE__);
    }
    return marker;
  }
  /************************************
 *
 *
 *
 *
 ************************************/

  cv::Mat  FiducialMarkers::createBoardImage( Size gridSize,int MarkerSize,int MarkerDistance,  BoardConfiguration& TInfo  ,vector<int> *excludedIds) throw (cv::Exception)
  {



    srand(cv::getTickCount());
    int nMarkers=gridSize.height*gridSize.width;
    TInfo.resize(nMarkers);
    vector<int> ids=getListOfValidMarkersIds_random(nMarkers,excludedIds);
    for (int i=0;i<nMarkers;i++)
      TInfo[i].id=ids[i];

    int sizeY=gridSize.height*MarkerSize+(gridSize.height-1)*MarkerDistance;
    int sizeX=gridSize.width*MarkerSize+(gridSize.width-1)*MarkerDistance;
    //find the center so that the ref systeem is in it
    int centerX=sizeX/2;
    int centerY=sizeY/2;

    //indicate the data is expressed in pixels
    TInfo.mInfoType=BoardConfiguration::PIX;
    Mat tableImage(sizeY,sizeX,CV_8UC1);
    tableImage.setTo(Scalar(255));
    int idp=0;
    for (int y=0;y<gridSize.height;y++)
      for (int x=0;x<gridSize.width;x++,idp++) {
        Mat subrect(tableImage,Rect( x*(MarkerDistance+MarkerSize),y*(MarkerDistance+MarkerSize),MarkerSize,MarkerSize));
        Mat marker=createMarkerImage( TInfo[idp].id,MarkerSize);
        //set the location of the corners
        TInfo[idp].resize(4);
        TInfo[idp][0]=cv::Point3f( x*(MarkerDistance+MarkerSize),y*(MarkerDistance+MarkerSize),0);
        TInfo[idp][1]=cv::Point3f( x*(MarkerDistance+MarkerSize)+MarkerSize,y*(MarkerDistance+MarkerSize),0);
        TInfo[idp][2]=cv::Point3f( x*(MarkerDistance+MarkerSize)+MarkerSize,y*(MarkerDistance+MarkerSize)+MarkerSize,0);
        TInfo[idp][3]=cv::Point3f( x*(MarkerDistance+MarkerSize),y*(MarkerDistance+MarkerSize)+MarkerSize,0);
        for (int i=0;i<4;i++) TInfo[idp][i]-=cv::Point3f(centerX,centerY,0);
        marker.copyTo(subrect);
      }

    return tableImage;
  }

  /************************************
 *
 *
 *
 *
 ************************************/
  cv::Mat  FiducialMarkers::createBoardImage_ChessBoard( Size gridSize,int MarkerSize,  BoardConfiguration& TInfo ,bool centerData ,vector<int> *excludedIds) throw (cv::Exception)
  {


    srand(cv::getTickCount());

    //determine the total number of markers required
    int nMarkers= 3*(gridSize.width*gridSize.height)/4;//overdetermine  the number of marker read
    vector<int> idsVector=getListOfValidMarkersIds_random(nMarkers,excludedIds);


    int sizeY=gridSize.height*MarkerSize;
    int sizeX=gridSize.width*MarkerSize;
    //find the center so that the ref systeem is in it
    int centerX=sizeX/2;
    int centerY=sizeY/2;

    Mat tableImage(sizeY,sizeX,CV_8UC1);
    tableImage.setTo(Scalar(255));
    TInfo.mInfoType=BoardConfiguration::PIX;
    unsigned int CurMarkerIdx=0;
    for (int y=0;y<gridSize.height;y++) {

      bool toWrite;
      if (y%2==0) toWrite=false;
      else toWrite=true;
      for (int x=0;x<gridSize.width;x++) {
        toWrite=!toWrite;
        if (toWrite) {
          if (CurMarkerIdx>=idsVector.size()) throw cv::Exception(999," FiducialMarkers::createBoardImage_ChessBoard","INTERNAL ERROR. REWRITE THIS!!",__FILE__,__LINE__);
          TInfo.push_back( MarkerInfo(idsVector[CurMarkerIdx++]));

          Mat subrect(tableImage,Rect( x*MarkerSize,y*MarkerSize,MarkerSize,MarkerSize));
          Mat marker=createMarkerImage( TInfo.back().id,MarkerSize);
          //set the location of the corners
          TInfo.back().resize(4);
          TInfo.back()[0]=cv::Point3f( x*(MarkerSize),y*(MarkerSize),0);
          TInfo.back()[1]=cv::Point3f( x*(MarkerSize)+MarkerSize,y*(MarkerSize),0);
          TInfo.back()[2]=cv::Point3f( x*(MarkerSize)+MarkerSize,y*(MarkerSize)+MarkerSize,0);
          TInfo.back()[3]=cv::Point3f( x*(MarkerSize),y*(MarkerSize)+MarkerSize,0);
          if (centerData) {
            for (int i=0;i<4;i++)
              TInfo.back()[i]-=cv::Point3f(centerX,centerY,0);
          }
          marker.copyTo(subrect);
        }
      }
    }

    return tableImage;
  }



  /************************************
 *
 *
 *
 *
 ************************************/
  cv::Mat  FiducialMarkers::createBoardImage_Frame( Size gridSize,int MarkerSize,int MarkerDistance, BoardConfiguration& TInfo ,bool centerData,vector<int> *excludedIds ) throw (cv::Exception)
  {
    srand(cv::getTickCount());
    int nMarkers=2*gridSize.height*2*gridSize.width;
    vector<int> idsVector=getListOfValidMarkersIds_random(nMarkers,excludedIds);

    int sizeY=gridSize.height*MarkerSize+MarkerDistance*(gridSize.height-1);
    int sizeX=gridSize.width*MarkerSize+MarkerDistance*(gridSize.width-1);
    //find the center so that the ref systeem is in it
    int centerX=sizeX/2;
    int centerY=sizeY/2;

    Mat tableImage(sizeY,sizeX,CV_8UC1);
    tableImage.setTo(Scalar(255));
    TInfo.mInfoType=BoardConfiguration::PIX;
    int CurMarkerIdx=0;
    int mSize=MarkerSize+MarkerDistance;
    for (int y=0;y<gridSize.height;y++) {
      for (int x=0;x<gridSize.width;x++) {
        if (y==0 || y==gridSize.height-1 || x==0 ||  x==gridSize.width-1) {
          TInfo.push_back(  MarkerInfo(idsVector[CurMarkerIdx++]));
          Mat subrect(tableImage,Rect( x*mSize,y*mSize,MarkerSize,MarkerSize));
          Mat marker=createMarkerImage( TInfo.back().id,MarkerSize);
          marker.copyTo(subrect);
          //set the location of the corners
          TInfo.back().resize(4);
          TInfo.back()[0]=cv::Point3f( x*(mSize),y*(mSize),0);
          TInfo.back()[1]=cv::Point3f( x*(mSize)+MarkerSize,y*(mSize),0);
          TInfo.back()[2]=cv::Point3f( x*(mSize)+MarkerSize,y*(mSize)+MarkerSize,0);
          TInfo.back()[3]=cv::Point3f( x*(mSize),y*(mSize)+MarkerSize,0);
          if (centerData) {
            for (int i=0;i<4;i++)
              TInfo.back()[i]-=cv::Point3f(centerX,centerY,0);
          }

        }
      }
    }

    return tableImage;
  }
  /************************************
 *
 *
 *
 *
 ************************************/
  Mat FiducialMarkers::rotate(const Mat  &in)
  {
    Mat out;
    in.copyTo(out);
    for (int i=0;i<in.rows;i++)
    {
      for (int j=0;j<in.cols;j++)
      {
        out.at<uchar>(i,j)=in.at<uchar>(in.cols-j-1,i);
      }
    }
    return out;
  }
  /************************************
 *
 *
 *
 *
 ************************************/
  //http://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
  int32_t HammingWeight(int32_t i) {
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
  }

  /************************************
 *
 *
 *
 *
 ************************************/
  FiducialMarkers::MDR FiducialMarkers::hammDistMarker(Mat  bits, int gsize) {
    std::vector<int> markers;
    // For other sizes just figure out a marker id dictionary and fill in the vector
    if(gsize == 3) markers = {2,31,69,107,118,167,186,206,211,253};
    if(markers.size() > 0) {
      int n = 0;
      for(int y = 0; y < gsize; y++)
        for(int x = 0; x < gsize; x++)
          if(bits.at<uchar>(y,x) == 1)
            n += (1 << y * gsize + x);

      int minDist = -1;
      MDR mdr;
      for(int i = 0; i < markers.size(); i++) {
        int32_t dist = HammingWeight(markers[i] ^ n);
        if(mdr.marker == -1 || dist < mdr.dist) {
          mdr.marker = markers[i];
          mdr.dist = dist;
        }
      }
      return mdr;
    } else if(gsize == 5) {
      int ids[4][5]= {{1,0,0,0,0},
                      {1,0,1,1,1},
                      {0,1,0,0,1},
                      {0,1,1,1,0}};
      int dist=0;
      for (int y=0;y<gsize;y++) {
        int minSum=1e5;
        //hamming distance to each possible word
        for (int p=0;p<4;p++) {
          int sum=0;
          //now, count
          for (int x=0;x<gsize;x++)
            sum+=  bits.at<uchar>(y,x) == ids[p][x]?0:1;
          if (minSum>sum) minSum=sum;
        }
        //do the and
        dist+=minSum;
      }
      MDR ret;
      ret.dist = dist;
      return ret;
    } else throw cv::Exception(9190,"Invalid marker grid size","aruco::fiducial::hammDistMarker",__FILE__,__LINE__);
  }

  /************************************
 *
 *
 *
 *
 ************************************/
  int FiducialMarkers::analyzeMarkerImage(Mat &grey,int &nRotations, int gsize)
  {
    //G == gsize, B = bsize
    //Markers  are divided in BxB regions, of which the inner GxG belongs to marker info
    //the external border shoould be entirely black

    int bsize = gsize + 2;
    int swidth=grey.rows/bsize;
    for (int y=0;y<bsize;y++)
    {
      int inc=bsize - 1;
      if (y==0 || y==bsize-1) inc=1;//for first and last row, check the whole border
      for (int x=0;x<bsize;x+=inc)
      {
        int Xstart=(x)*(swidth);
        int Ystart=(y)*(swidth);
        Mat square=grey(Rect(Xstart,Ystart,swidth,swidth));
        int nZ=countNonZero(square);
        if (nZ> (swidth*swidth) /2) {
          // 		cout<<"neb"<<endl;
          return -1;//can not be a marker because the border element is not black!
        }
      }
    }

    //now,
    vector<int> markerInfo(gsize);
    Mat _bits=Mat::zeros(gsize,gsize,CV_8UC1);
    //get information(for each inner square, determine if it is  black or white)

    for (int y=0;y<gsize;y++)
    {

      for (int x=0;x<gsize;x++)
      {
        int Xstart=(x+1)*(swidth);
        int Ystart=(y+1)*(swidth);
        Mat square=grey(Rect(Xstart,Ystart,swidth,swidth));
        int nZ=countNonZero(square);
        if (nZ> (swidth*swidth) /2)  _bits.at<uchar>( y,x)=1;
      }
    }
    // 		printMat<uchar>( _bits,"or mat");

    //checkl all possible rotations
    Mat _bitsFlip;
    Mat Rotations[4];
    Rotations[0]=_bits;
    MDR mdrs[4];
    int dists[4];
    mdrs[0] = hammDistMarker(Rotations[0], gsize);
    MDR minimum = mdrs[0]; 
    minimum.rotation = 0;
    for (int i=1;i<4;i++) {
      //rotate
      Rotations[i] = rotate(Rotations[i-1]);
      //get the hamming distance to the nearest possible word
      mdrs[i] = hammDistMarker(Rotations[i], gsize);
      if(mdrs[i].dist < minimum.dist) {
        minimum = mdrs[i];
        minimum.rotation = i;
      }
    }
    nRotations = minimum.rotation;
    if(minimum.dist > 0 && gsize == 5) return -1;
    // 3 is the minimum pairwise distance for the optimal 
    // marker configuration on gsize = 3, n markers = 10
    if(minimum.dist >= 3) return -1; 

    cv::Mat bits = Rotations[minimum.rotation];
    if(gsize == 5) {//Get id of the marker
      int MatID=0;
      for (int y=0;y<gsize;y++)
      {
        MatID<<=1;
        if ( bits.at<uchar>(y,1)) MatID|=1;
        MatID<<=1;
        if ( bits.at<uchar>(y,3)) MatID|=1;
      }
      return MatID;
    } else {
      return minimum.marker;
    }
  }


  /************************************
 *
 *
 *
 *
 ************************************/
  int FiducialMarkers::detect(const Mat &in,int &nRotations)
  {
    assert(in.rows==in.cols);
    Mat grey;
    if ( in.type()==CV_8UC1) grey=in;
    else cv::cvtColor(in,grey,CV_BGR2GRAY);
    //threshold image
    threshold(grey, grey,125, 255, THRESH_BINARY|THRESH_OTSU);

    //now, analyze the interior in order to get the id
    //try first with the big ones

    return analyzeMarkerImage(grey,nRotations);;
    //too many false positives
    /*    int id=analyzeMarkerImage(grey,nRotations);
        if (id!=-1) return id;
        id=analyzeMarkerImage_type2(grey,nRotations);
        if (id!=-1) return id;
        return -1;*/
  }

  vector<int> FiducialMarkers::getListOfValidMarkersIds_random(int nMarkers,vector<int> *excluded) throw (cv::Exception)
  {

    if (excluded!=NULL)
      if (nMarkers+excluded->size()>1024) throw cv::Exception(8888,"FiducialMarkers::getListOfValidMarkersIds_random","Number of possible markers is exceeded",__FILE__,__LINE__);

    vector<int> listOfMarkers(1024);
    //set a list with all ids
    for (int i=0;i<1024;i++) listOfMarkers[i]=i;

    if (excluded!=NULL)//set excluded to -1
      for (size_t i=0;i<excluded->size();++i)
        listOfMarkers[excluded->at(i)]=-1;
    //random shuffle
    random_shuffle(listOfMarkers.begin(),listOfMarkers.end());
    //now, take the first  nMarkers elements with value !=-1
    int i=0;
    vector<int> retList;
    while (static_cast<int>(retList.size())<nMarkers) {
      if (listOfMarkers[i]!=-1)
        retList.push_back(listOfMarkers[i]);
      ++i;
    }
    return retList;
  }

}

