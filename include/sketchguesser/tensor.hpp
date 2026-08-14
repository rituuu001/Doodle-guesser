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
    float& operator[](size_t idx){return data[idx];}
    
    //read only
    const float& operator[](size_t idx) const { return data[idx];}

    std::vector<float>& getData() { return data;}
    const std::vector<float>& getData() const { return data;}

    size_t size() const { return data.size();}
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
  
};