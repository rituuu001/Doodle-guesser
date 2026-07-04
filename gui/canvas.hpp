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
       
        // For later (Week 5) — read-only access to the raw buffer
        //const std::vector<unsigned char>& getBuffer() const;
        int getWidth() const;
        int getHeight() const;

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

