#pragma once
#include <vector>
#include <raylib.h>


    class Canvas {
    public:
        Canvas(int bufferWidth, int bufferHeight, int brushRadius);

        // Called once per frame by Ui, with mouse position already
        // converted into canvas-local coordinates (0,0 = top-left of canvas)
        void handleInput(Vector2 localMousePos, bool mouseDown);

        // Resets the buffer to blank (0)
        void clear();

        // Renders the current buffer to screen at the given top-left position
        void draw(Vector2 screenPosition) const;
       
        const std::vector<unsigned char>& getBuffer() const
        {
            return buffer_;
        }

        int getWidth() const
        {
            return width_;
        }
        
        int getHeight() const
        {
            return height_;
        }

    private:
        int width_;
        int height_;
        int brushRadius_;
        std::vector<unsigned char> buffer_;

        bool wasDrawingLastFrame_;
        Vector2 lastPoint_;

        // your stroke-painting logic goes here
        void paintPoint(Vector2 point);
        void paintLine(Vector2 from, Vector2 to);
       
    };

