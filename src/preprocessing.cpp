#include "sketchguesser/preprocessing.hpp"
#include<stdexcept>
#include<opencv2/opencv.hpp>

DoodleGuesser:: BoundingBox DoodleGuesser:: findBoundingBox(const uint8_t* buffer, int width, int height)
{
  DoodleGuesser:: BoundingBox bb;
  bb.minX=width;
  bb.maxX=0;
  bb.minY=height;
  bb.maxY=0;
  for (int x=0;x<width;x++)
  {
    for(int y=0;y<height;y++)
    {
        int index=x+y*width;
        if (buffer[index]>0)
        {
            bb.maxX=x>bb.maxX?x:bb.maxX;
            bb.minX=x<bb.minX?x:bb.minX;
            bb.maxY=y>bb.maxY?y:bb.maxY;
            bb.minY=y<bb.minY?y:bb.minY;

        }
    }
  }
  if (bb.minX>bb.maxX || bb.minY>bb.maxY)
  throw std::runtime_error("Nothing has been drawn");
  return bb;
}

Tensor DoodleGuesser:: preprocess(const uint8_t* buffer, int width, int height)
{
  //CV_8UC1 8 but unsigned int,channel 1
  //image formed from buffer
  cv::Mat img(height, width, CV_8UC1, const_cast<uint8_t*>(buffer));
  BoundingBox bb=findBoundingBox(buffer, width, height);
  //crops the empty space
  cv::Mat cropped=img(cv::Rect(bb.minX, bb.minY, bb.maxX-bb.minX+1, bb.maxY-bb.minY+1));
  //scaling of the doodle
  int larger=cropped.cols>cropped.rows?cropped.cols:cropped.rows;
  float scale=28.0/larger;
  int newWidth=std::round(cropped.cols*scale);
  int newHeight=std::round(cropped.rows*scale);
  
  cv::Mat destination;
  cv::resize(cropped, destination, cv::Size(newWidth, newHeight));

  cv::Mat finalCanvas(28, 28, CV_8UC1, cv::Scalar(0));
  //Centering of the scaled doodle
  int xOffset=(28-newWidth)/2;
  int yOffset=(28-newHeight)/2;

  destination.copyTo(finalCanvas(cv::Rect(xOffset, yOffset, newWidth, newHeight)));
  
  Tensor final(1,28,28);

  for (int col=0;col<28;col++)
  {
    for (int row=0;row<28;row++)
    {
     uchar pixel = finalCanvas.at<uchar>(row, col);
     //normalized the pixel value 0.0-1.0 
     float normalized=pixel/255.0;
     final(0,row,col)=normalized;
    }
  } 
 
  return final;
}