#include "raylib.h"
#include "gui/canvas.hpp"

int main()
{
    InitWindow(400, 300, "Canvas Test");
    Canvas canvas(280, 280, 5);

    while (!WindowShouldClose())
    {
        Vector2 pos = GetMousePosition();
        Vector2 localPos = { pos.x - 10, pos.y - 10 };
        bool mouseDown = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
        canvas.handleInput(localPos, mouseDown);

        BeginDrawing();
        ClearBackground(RAYWHITE);
        canvas.draw({ 10, 10 });
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
  