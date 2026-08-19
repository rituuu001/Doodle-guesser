# Sketch-Guesser

A real-time hand-drawn sketch recognition system powered by a Convolutional
Neural Network built **entirely from scratch in C++17** — no external
machine learning frameworks. Draw on a canvas, and the model classifies
your sketch into one of six categories in real time.

This project was built as a learning exercise to understand the mechanics
that deep learning frameworks like PyTorch and TensorFlow normally
abstract away: the forward pass, backpropagation, loss computation, and
gradient-based optimization, all implemented by hand.


[Doodle-Guesser demo](docs/Screenshot 2026-08-14 215012.png)

## Overview

- Convolutional Neural Network — forward pass, backpropagation, loss, and
  SGD optimizer all implemented from scratch in C++17
- Trained on six categories sampled from Google's [Quick Draw
  dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- Real-time drawing canvas built with [Raylib](https://www.raylib.com/)
- Live input preprocessing (centering, resizing, normalization) using
  [OpenCV](https://opencv.org/)
- [Eigen](https://eigen.tuxfamily.org/) used for underlying matrix/vector
  math
- Python used **only** for offline dataset preparation — no Python
  dependency at runtime or in the core ML engine

## Architecture

The model classifies sketches into six categories:

**cup · donut · envelope · house · guitar · pants**

| Layer | Configuration | Output Shape | Trainable Params |
|---|---|---|---|
| Input | grayscale canvas | 1×28×28 | 0 |
| Conv1 | 4 filters, 3×3, stride 1 | 4×26×26 | 40 |
| ReLU1 | max(0, x) | 4×26×26 | 0 |
| MaxPool1 | 2×2, stride 2 | 4×13×13 | 0 |
| Flatten | — | 676 | 0 |
| FC1 | dense | 32 | 21,664 |
| ReLU2 | max(0, x) | 32 | 0 |
| FC2 | dense | 6 | 198 |
| Softmax | — | 6 | 0 |

**Total: 21,902 parameters.** Trained with categorical cross-entropy loss
and stochastic gradient descent (SGD, learning rate 0.01, 30 epochs by
default), entirely on CPU — no GPU acceleration, SIMD, or
multithreading.

## How it works

The project has two independent pipelines, sharing the same core network
code:

**Training pipeline** (offline)
```
Quick Draw dataset → Python preprocessor → dataset.bin →
C++ training loop (forward → loss → backward → SGD) → saved model weights
```

**Application pipeline** (runtime)
```
Raylib canvas → OpenCV preprocessing → trained network forward pass →
predicted class label
```

Every layer's backward pass was verified with gradient checking —
comparing analytically computed gradients against numerical
finite-difference estimates — before being trusted in training.

## Results

Trained for 30 epochs on an 80/20 train/validation split (4,800 / 1,200
images):

- Training accuracy: ~100%
- Validation accuracy: **~94%**
- Best-performing epoch is checkpointed automatically during training
  (validation accuracy peaks around epoch 16–18, after which the model
  begins to overfit)

 [Training curves](docs/cnntraining_and_validationaccuracy.png)

## Libraries Required

- **Eigen** (header-only) — vendored as a git submodule at
  `libraries/eigen`. [gitlab.com/libeigen/eigen](https://gitlab.com/libeigen/eigen)
- **Raylib 5.x** — vendored as a git submodule at `libraries/raylib`.
  [github.com/raysan5/raylib](https://github.com/raysan5/raylib)
- **OpenCV 4.x** — not bundled, must be installed separately.
  [opencv.org/releases](https://opencv.org/releases/) ·
  [installation guide](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html)
- **CMake 3.20+** — [cmake.org/download](https://cmake.org/download/)
- **Python 3.x with NumPy** — only needed if regenerating the dataset via
  `scripts/preprocess/`; not required to build or run the C++ project

## Building

1. Clone the repository with submodules:
   ```
   git clone --recurse-submodules <repository-url>
   ```
   If already cloned without submodules:
   ```
   git submodule update --init --recursive
   ```

2. Install OpenCV 4.x on your system, and make sure CMake's
   `find_package(OpenCV REQUIRED)` can locate it (via a package manager,
   or by setting the `OpenCV_DIR` environment variable).


3. Configure and build:
   ```
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build .
   ```
   *(Release mode is recommended for actual use — it runs noticeably
   faster than a Debug build, especially for training. Omit
   `-DCMAKE_BUILD_TYPE=Release` for a Debug build instead.)*

This produces:
- `train` — trains the CNN and saves the model
- `predict` — launches the GUI application for live sketch prediction
- `test_*` — unit and gradient-check tests

## Usage

A pre-trained model is already included in the repository, so **training
is optional** — you can jump straight to running the app.

**To try the model**
```
cmake --build . --target run_predict
```
Opens the drawing canvas using the included pre-trained model
(`../models/doodle_model.bin`, relative to the build directory). Draw a
sketch, click Predict, and the model's guess is displayed on screen.

**To train the model**
1.  Download raw QuickDraw numpy bitmaps (into `data/raw/`):

   ```bash
    python3 scripts/preprocess/download_quickdraw.py
   ```
   Pack sampled images into a compact binary dataset:
   ```bash
    python3 scripts/preprocess/pack_dataset.py
   ```
2. Train
```
cmake --build . --target run_train
```
Reads the dataset from `../data/processed/dataset.bin` (relative to the
build directory), trains for 30 epochs by default (learning rate 0.01),
logs training/validation loss and accuracy to CSV, and saves the
best-performing model — this will overwrite the included one.

**Run the test suite:**
```
cmake --build . --target run_tests
```

*(Each target above can also be run as its own executable directly from
the build directory, e.g. `./train`, `./predict`, if you prefer.)*

> **Windows note:** the build automatically copies the required OpenCV
> DLL next to the `predict` executable. If it's missing at runtime, check
> that the OpenCV DLL path in `CMakeLists.txt` matches your local OpenCV
> installation.

## Project Structure

```
CMakeLists.txt
libraries/eigen, libraries/raylib      — vendored dependencies
include/sketchguesser/                 — headers (Tensor, layers, network,
                                          loss, optimizer, dataset,
                                          serializer, preprocessing)
src/                                   — implementations, train_main.cpp,
                                          app_main.cpp
gui/                                   — Raylib canvas and UI
scripts/preprocess/                    — Python dataset download/packing
tests/                                 — unit tests and gradient checks
```

## Scope

**In scope:** a 4-filter, single-convolution CNN across six sketch
categories; full from-scratch forward/backward implementation; a desktop
GUI with live drawing and prediction; binary model serialization.

**Out of scope:** GPU/SIMD/multithreaded training; deeper architectures
(additional conv layers, dropout, batch normalization); mobile or web
deployment; advanced runtime statistics (confidence histograms, top-5
predictions).

## Future Enhancements

- Adaptive optimizers (Adam, RMSProp) in place of plain SGD
- Mini-batch gradient descent to reduce overfitting
- Data augmentation (rotation, translation, scaling)
- Per-class precision/recall and confusion matrix evaluation
- Expanded category vocabulary
- GPU/SIMD acceleration

## References

1. Google Creative Lab. (2017). *Quick, Draw! Dataset.*
   [github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based
   learning applied to document recognition.* Proceedings of the IEEE,
   86(11), 2278–2324.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.*
   MIT Press.
4. Raylib Development Team. (2024). *Raylib 5.0 Architecture and
   Documentation.* [raylib.com](https://www.raylib.com)
5. OpenCV Open Source Team. (2024). *OpenCV 4.x Reference Manual.*
   [docs.opencv.org](https://docs.opencv.org)

