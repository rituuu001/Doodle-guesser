#include"canvas.hpp"
#include<cmath>
#include <algorithm>
#include<raylib.h>


Canvas::Canvas(int bufferWidth, int bufferHeight, int brushRadius) :
	width_(bufferWidth),
    height_(bufferHeight),
	brushRadius_(brushRadius),
	buffer_(width_* height_, 0),
	wasDrawingLastFrame_(false),
	lastPoint_({ 0, 0 }) {}

void Canvas::paintPoint(Vector2 point)
{   
	int x = static_cast<int>(point.x);
	int y = static_cast<int>(point.y);
	
	int maxX = x + brushRadius_;
	int minX = x- brushRadius_;
	int maxY = y + brushRadius_;
	int minY = y - brushRadius_;
	for (int py = minY;py <= maxY;py++)
	{
		for (int px = minX;px <= maxX;px++)
		{
			if (px < 0 || py < 0 || px >= width_ || py >= height_)
				continue;
			int dx = px - x;
			int dy = py - y;
			int sqrDist = dx * dx + dy * dy;
			if (sqrDist < brushRadius_ * brushRadius_)
			{
				//index of buffer_
				int index = px + py * width_;
				buffer_[index] = 255;
			}


		}
	}
	
}
void Canvas::paintLine(Vector2 from, Vector2 to)
{

	float dx = to.x - from.x;
	float dy = to.y - from.y;
	float dist = std::sqrt(dx * dx + dy * dy);
	float unitf = std::ceil(dist / brushRadius_);
	int units = static_cast<int>(unitf);
	if (dx != 0 || dy != 0)
	{
		for (int count = 0;count <= units;count++)
		{
			float fraction = count / unitf;
			float x = from.x + fraction * (to.x - from.x);
			float y = from.y + fraction * (to.y - from.y);
			Vector2 pos = { x,y };
			paintPoint(pos);

		}
	}
	else
	{
		paintPoint(to);
	}
	
}

void Canvas::handleInput(Vector2 localMousePos, bool mouseDown)
{
	
	if (mouseDown)
	{
		if (!wasDrawingLastFrame_)
		{
			paintPoint(localMousePos);
		}
		else
		{
			paintLine(lastPoint_, localMousePos);
		}	
		lastPoint_ = localMousePos;
		wasDrawingLastFrame_ = true;
	}
	else
	{
		wasDrawingLastFrame_ = false;
	}
}
void Canvas::draw(Vector2 screenPosition) const
{
	for (int y = 0; y < height_; y++) {
		for (int x = 0; x < width_; x++) {
			int index = x + y * width_;
			if (buffer_[index] > 0) {
				int screenX = screenPosition.x + x;
				int screenY = screenPosition.y + y;
				DrawPixel(screenX, screenY, BLACK);
			}
		}
	}
}
void Canvas::clear()
{
	std::fill(buffer_.begin(), buffer_.end(), 0);
}