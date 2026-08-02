#pragma once
#include <raylib.h>
#include <string>
#include "canvas.hpp"

#include "sketchguesser/predictor.hpp"

const Color MUSTARD = { 244, 180, 0, 255 };
class Predictor; 
class UI
{
public:
    UI(Predictor& predictor);
    ~UI();

    void draw() const;
    void handleInput(Canvas& canvas);
   

private:
    Rectangle canvasBorderRect_;
    Rectangle clearButtonRect_;
    Rectangle predictButtonRect_;
    std::string predictionText_;
    Font fontTitle,fontBody;
    Texture2D predictIcon,clearIcon,logoIcon;

    Predictor& predictor_;
};
