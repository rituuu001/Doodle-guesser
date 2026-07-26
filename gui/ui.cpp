#include"ui.hpp"
#include"canvas.hpp"

UI::UI()
{   
    canvasBorderRect_={250, 150, 336, 336};
    predictButtonRect_={200, 600, 200, 50};
    clearButtonRect_={450, 600, 200, 50};
    

    predictionText_="Prediction:-";

    predictIcon = LoadTexture("gui/assets/icons/predict.png");
    clearIcon = LoadTexture("gui/assets/icons/clear.png");
    logoIcon = LoadTexture("gui/assets/icons/logo.png");

    fontBody=LoadFontEx("gui/assets/fonts/PlusJakartaSans-Regular.ttf", 40, nullptr, 0);
    fontTitle=LoadFontEx("gui/assets/fonts/Manrope-ExtraBold.ttf", 60, nullptr, 0);
}

UI::~UI()
{
    UnloadTexture(predictIcon);
    UnloadTexture(clearIcon);
    UnloadTexture(logoIcon);
    UnloadFont(fontBody);
    UnloadFont(fontTitle);
}

void UI:: draw() const
{
    DrawCircle(20, 20, 8, Color{ 255, 95, 86, 255 });   // red
    DrawCircle(45, 20, 8, Color{ 255, 189, 46, 255 });  // yellow
    DrawCircle(70, 20, 8, Color{ 39, 201, 63, 255 });   // green

    DrawTextEx(fontTitle, "Doodle Guesser", { 250, 50 }, 60, 1, BLACK);
    DrawTextureEx(logoIcon,{150,50}, 0.0f, 0.25f, WHITE);

    DrawRectangleRoundedLinesEx(canvasBorderRect_, 0.05f, 8, 4, MUSTARD);

    DrawTextEx(fontBody, predictionText_.c_str(), { 250, 520 }, 40, 1, GRAY);

    DrawRectangleRounded(predictButtonRect_, 0.5f, 8, MUSTARD);
    DrawTextEx(fontBody, "Predict", { predictButtonRect_.x + 25, predictButtonRect_.y + 5 }, 40, 1, WHITE);
    DrawTextureEx(predictIcon,{predictButtonRect_.x + 150, predictButtonRect_.y + 10}, 0.0f, 0.1f, WHITE);

    DrawRectangleRounded(clearButtonRect_, 0.5f, 8, MUSTARD);
    DrawTextEx(fontBody, "Clear", { clearButtonRect_.x + 25, clearButtonRect_.y + 5 }, 40, 1, WHITE);
    DrawTextureEx(clearIcon,{clearButtonRect_.x + 150, clearButtonRect_.y + 10}, 0.0f, 0.1f, WHITE);
}


void UI::handleInput(Canvas& canvas)
{
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    {
        Vector2 mouse = GetMousePosition();

        if (CheckCollisionPointRec(mouse, clearButtonRect_))
        {
            canvas.clear();
        }
        // Predict button: no logic yet, per Task 1 scope
    }
}
 