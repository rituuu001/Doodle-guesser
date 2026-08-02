#include <raylib.h>
#include "../gui/canvas.hpp"
#include "../gui/ui.hpp"


int main()
{
    InitWindow(850, 750, "Sketch Guesser - Test");
    SetTargetFPS(60);

    Canvas canvas(336, 336, 6);
    UI ui;  // safe: constructed after InitWindow, so GPU context exists

    Vector2 canvasScreenPos = { 250, 150 };

    while (!WindowShouldClose())
    {
        Vector2 mouse = GetMousePosition();
        Vector2 localMouse = { mouse.x - canvasScreenPos.x, mouse.y - canvasScreenPos.y };
        bool mouseDown = IsMouseButtonDown(MOUSE_LEFT_BUTTON);

        // only feed drawing input to canvas if mouse is inside its bounds
        bool insideCanvas = localMouse.x >= 0 && localMouse.y >= 0 &&
                             localMouse.x < canvas.getWidth() && localMouse.y < canvas.getHeight();

        canvas.handleInput(localMouse, mouseDown && insideCanvas);
        ui.handleInput(canvas);

        BeginDrawing();
        ClearBackground(Color {255, 249, 232});

        canvas.draw(canvasScreenPos);
        ui.draw();

        EndDrawing();
    }

    CloseWindow();
    return 0;
}