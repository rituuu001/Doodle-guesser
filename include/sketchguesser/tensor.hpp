#pragma once

#include<vector>
class Tensor{
private:
 int channels;
 int height;       //same as row
 int width;      //same as column      
 std::vector<float> data;  //to flatten the tensor to make it easier to store variable size
                                         // since we can change the size
public:
 //making a constructor so that every tensor start with a valid state
 Tensor(int c,int h,int w)
  : channels(c),height(h),width(w),data(c*h*w,0.0f)
  {}
  int getChannels() const{
    return channels;
  }
  int getHeight() const{
    return height;
  }
  int getWidth() const{
    return width;
  }
  int getDataSize() const{
    return data.size();
  }
 float getPixel(int c,int h,int w) const {
  int index=c*height*width+h*width+w;
  return data[index];
 }                            
};