#include <Eigen/Dense>
#include <raylib.h>

int main() {
    Eigen::Matrix2f m;
    m << 1, 2,
        3, 4;

    InitWindow(400, 300, "Test");

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawText("Libraries are working!", 50, 50, 20, BLACK);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}