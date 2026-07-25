#pragma once
#include<stdexcept> // for throw std::
#include<vector>
class Tensor{
private:
 int channels;
 int height;       //same as row
 int width;      //same as column      
 std::vector<float> data;  //to flatten the tensor to make it easier to store variable size
                                         // since we can change the size
 //making a constructor so that every tensor start with a valid state
 public:
    // Default constructor
    Tensor()
        : channels(0), height(0), width(0), data()
    {}

    // Constructor with dimensions
    Tensor(int c, int h, int w)
        : channels(c), height(h), width(w), data(c * h * w, 0.0f)
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
  int size() const{
    return data.size();
  }
 float& operator()(int c,int h,int w)  {
   if(c < 0 || c >= channels ||
       h < 0 || h >= height ||
       w < 0 || w >= width)
    {
        throw std::out_of_range("Tensor index out of range");
    }
  int index=c*height*width+h*width+w;
  return data[index];
 }                            
 const float& operator()(int c,int h,int w) const
{
    if(c < 0 || c >= channels ||
       h < 0 || h >= height ||
       w < 0 || w >= width)
    {
        throw std::out_of_range("Tensor index out of range");
    }

    int index = c * height * width + h * width + w;
    return data[index];
}
// 1D access that can be changed also
float& operator()(int index)
{
    return data[index];
}

// 1D access that is read only
const float& operator()(int index) const
{
    return data[index];
}



};