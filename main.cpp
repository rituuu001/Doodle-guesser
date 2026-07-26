#include <vector>
#include <cstdint>
#include <iostream>
#include "sketchguesser/preprocessing.hpp"

int main()
{
    const int width = 336;
    const int height = 336;

    // Blank canvas, all background (0)
    std::vector<uint8_t> buffer(width * height, 0);

    for (int y = 100; y <= 160; y++)
    {
        for (int x = 10; x <= 310; x++)
        {
            buffer[x + y * width] = 255;
        }
    }

    std::cout << "Running preprocess() on synthetic 336x336 buffer with a WIDE rectangle at "
                 "x:[10,310] (301px wide), y:[100,160] (61px tall)\n";
    std::cout << "Expected bounding box: minX=10, maxX=310, minY=100, maxY=160\n";
    std::cout << "Width is the larger dimension -> after resize, width should be ~28, "
                 "height should be noticeably smaller than 28 (padded top/bottom)\n";

    Tensor result = preprocess(buffer.data(), width, height);

    std::cout << "Tensor produced: channels=" << result.getChannels()
              << " height=" << result.getHeight()
              << " width=" << result.getWidth() << "\n";
    std::cout << "Check debug_output.png (written inside preprocess(), if you added "
                 "the temporary cv::imwrite line) to visually verify centering.\n";

    return 0;
}