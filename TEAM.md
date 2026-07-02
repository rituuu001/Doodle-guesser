# Sketch-Guesser — Team Notes

Internal notes for Ritu, Samir, and Samita. Not meant for the supervisor-facing
`README.md` — this is our working doc.

## Roles (rotate weekly — see table below)

| Person | 
|---|---|---|
| Ritu Adhikari | | |
| Samir Neupane | | |
| Samita Regmi | | |

## Task Division — Full Timeline

| Week | Person 1  | Person 2  | Person 3  |
|---|---|---|---|
| 1 — Setup & Dataset | CMake setup; link Eigen/Raylib/OpenCV; C++ dataset loader stub | Python downloader + 1,000/class sampler | Binary packer design + writer; start Raylib window |
| 2 — Forward Pass & UI | `Layer` base + Conv `forward()` | ReLU + MaxPool `forward()` | FC + Softmax `forward()`; drawing canvas |
| 3 — Backprop & Loss | Conv `backward()` + gradient check | FC `backward()` + gradient check | Pool/ReLU `backward()` + cross-entropy loss + gradient check |
| 4 — Training & Storage | SGD optimizer + training loop | Wire loop to dataset; first convergence run | Binary weight save/load |
| 5 — Integration | Tune training convergence | Canvas → OpenCV preprocessing | OpenCV → inference engine → label display |
| 6 — Testing & Docs | Unit tests (Conv/Pool); latency profiling | Unit tests (FC/Softmax/loss); confusion matrix | Report + slides |

**Week 3 rule:** everyone walks the other two through their own gradient derivation before merging. No solo backprop.
**Week 6 rule:** write up the section you did *not* build.

## File Guide — What Each File Does

### Root
| File | Purpose |
|---|---|
| `CMakeLists.txt` | Build configuration — links Eigen, Raylib, OpenCV; defines the two executables (`train_main`, `app_main`) and the test binary. |
| `main.cpp` | Entry point (if used as a shared/top-level entry — confirm whether this stays or gets replaced fully by `train_main.cpp`/`app_main.cpp`). |
| `.gitmodules` | Tracks the Eigen and Raylib submodules and their source URLs. |
| `.gitignore` | Keeps build artifacts (`build/`), generated datasets (`.bin` files), and IDE files out of the repo. |
| `README.md` | Supervisor/outside-facing overview — project description, build & run instructions. |
| `TEAM.md` | This file — internal team notes. |

### `libraries/`
| File | Purpose |
|---|---|
| `libraries/eigen` | Eigen submodule — header-only linear algebra library used throughout the network for matrix/vector math. |
| `libraries/raylib` | Raylib submodule — powers the drawing canvas and GUI window. |

### `include/sketchguesser/` (headers — class/function declarations)
| File | Purpose |
|---|---|
| `tensor.hpp` | Core `Tensor` class — wraps multi-dimensional data (e.g. channels × height × width) that every layer passes between forward/backward calls. Everything else depends on this being right. |
| `layers/layer.hpp` | Abstract base `Layer` class — declares the `forward()`/`backward()` interface every layer type implements (enables polymorphic layer chaining in `Network`). |
| `layers/conv_layer.hpp` | Convolution layer declaration — 4 filters, 3×3 kernel, stride 1. |
| `layers/relu_layer.hpp` | ReLU activation layer declaration. |
| `layers/maxpool_layer.hpp` | 2×2 max pooling layer declaration. |
| `layers/fc_layer.hpp` | Fully connected (dense) layer declaration. |
| `layers/softmax_layer.hpp` | Softmax output layer declaration. |
| `network.hpp` | `Network` class — holds an ordered sequence of `Layer` objects, chains forward passes and backward passes across the full model. |
| `loss.hpp` | Categorical cross-entropy loss function declaration. |
| `optimizer.hpp` | SGD optimizer declaration — applies gradients to update layer weights. |
| `dataset.hpp` | Declares the dataset loader — reads `dataset.bin`, exposes images/labels to the training loop. |
| `serializer.hpp` | Declares binary read/write functions for saving and loading trained model weights. |
| `preprocessing.hpp` | Declares image preprocessing functions — centering, resizing to 28×28, normalization (used both at training-data-prep time conceptually and at live-inference time via OpenCV). |
| `utils.hpp` | Shared helper functions/utilities used across multiple files (e.g. random init, logging). |

### `src/` (implementations)
| File | Purpose |
|---|---|
| `tensor.cpp` | `Tensor` class implementation. |
| `layers/conv_layer.cpp` | Convolution forward/backward math. |
| `layers/maxpool_layer.cpp` | Max pooling forward/backward math. |
| `layers/fc_layer.cpp` | Fully connected forward/backward math. |
| `layers/softmax_layer.cpp` | Softmax forward + integration with cross-entropy backward. |
| *(no `relu_layer.cpp`)* | ReLU is simple enough it may be implemented inline in the header — confirm this is intentional. |
| `network.cpp` | Chains layers together; runs full forward pass and full backward pass across the network. |
| `loss.cpp` | Cross-entropy loss computation + gradient. |
| `optimizer.cpp` | SGD weight update logic. |
| `dataset.cpp` | Reads `dataset.bin` into memory, splits into train/val (80/20). |
| `serializer.cpp` | Writes/reads trained weights and biases to/from binary files. |
| `preprocessing.cpp` | OpenCV-based image preprocessing — used at inference time to convert canvas drawings into normalized 28×28 tensors. |
| `utils.cpp` | Helper function implementations. |
| `train_main.cpp` | Training executable entry point — loads dataset, builds `Network`, runs the training loop, saves weights. |
| `app_main.cpp` | Application executable entry point — loads saved weights, launches the GUI, runs live inference. |

### `gui/`
| File | Purpose |
|---|---|
| `canvas.hpp` / `canvas.cpp` | Drawing canvas logic — tracks mouse/cursor strokes into a pixel buffer using Raylib. |
| `ui.hpp` / `ui.cpp` | Overall window layout — Predict/Clear buttons, prediction label display, frame loop. |

### `scripts/preprocess/`
| File | Purpose |
|---|---|
| `download_quickdraw.py` | Downloads the six chosen Quick Draw categories (pre-rasterized `.npy` bitmaps) and samples 1,000 balanced images per class. |
| `pack_dataset.py` | Packs sampled images + labels into the custom `dataset.bin` binary format the C++ side reads. |

### `tests/`
| File | Purpose |
|---|---|
| `test_main.cpp` | Unit tests — layer forward/backward correctness, gradient checks, dataset round-trip verification. |

## Task Division by File — Full Timeline

| Week | Person 1 | Person 2 | Person 3 |
|---|---|---|---|
| **1 — Setup & Dataset** | `CMakeLists.txt`, `.gitmodules` setup, `tensor.hpp`/`tensor.cpp` skeleton, `dataset.hpp`/`dataset.cpp` loader stub | `scripts/preprocess/download_quickdraw.py` | `scripts/preprocess/pack_dataset.py`; start `gui/canvas.hpp`/`.cpp` window stub |
| **2 — Forward Pass & UI** | `layers/layer.hpp` (base class), `layers/conv_layer.hpp`/`.cpp` `forward()` | `layers/relu_layer.hpp`, `layers/maxpool_layer.hpp`/`.cpp` `forward()` | `layers/fc_layer.hpp`/`.cpp`, `layers/softmax_layer.hpp`/`.cpp` `forward()`; build out `gui/canvas.cpp` |
| **3 — Backprop & Loss** | `conv_layer.cpp` `backward()` + gradient check in `tests/test_main.cpp` | `fc_layer.cpp` `backward()` + gradient check | `maxpool_layer.cpp`/ReLU `backward()`, `loss.hpp`/`loss.cpp` (cross-entropy) + gradient check |
| **4 — Training & Storage** | `optimizer.hpp`/`.cpp` (SGD) + `train_main.cpp` loop skeleton | Wire `train_main.cpp` to `dataset.cpp`; run first convergence pass, log loss | `serializer.hpp`/`.cpp` — save/load weights |
| **5 — Integration** | Tune convergence (learning rate, epochs) in `train_main.cpp` | `gui/canvas.cpp` → `preprocessing.cpp` (OpenCV) wiring | `preprocessing.cpp` → `network.hpp`/`.cpp` forward pass → `gui/ui.cpp` label display |
| **6 — Testing & Docs** | Unit tests for Conv/Pool in `tests/test_main.cpp`; latency profiling (`std::chrono`) in `app_main.cpp` | Unit tests for FC/Softmax/loss; confusion matrix script | `README.md` + report + slides |

**Week 3 rule:** everyone walks the other two through their own gradient derivation before merging. No solo backprop.
**Week 6 rule:** write up the section you did *not* build.

## Repo Conventions

- **Branches:** _(fill in — e.g. one feature branch per person per week, PR into `main` before Friday sync)_
- **Commits:** _(fill in — e.g. `type: short description`, like `feat: add conv layer forward pass`)_
- **Formatting:** _(fill in — e.g. clang-format style, header guard convention)_

## Build Quirks / Gotchas

- Clone with `git clone --recurse-submodules ...` — or run
  `git submodule update --init --recursive` after a normal clone, otherwise
  `libraries/eigen` and `libraries/raylib` will be empty and the build fails.
- _(add OS-specific linking issues here as you hit them)_

## Decisions Log

- Using Google's pre-rasterized Quick Draw `.npy` bitmap format (not raw
  `.ndjson` strokes) — Python restricted to offline preprocessing only, per
  proposal.
- _(add future decisions here with date + reason, e.g. "changed learning rate
  from X to Y because...")_

## Weekly Sync Notes

### Week 1
- _(what got done, what's blocked, who's behind)_

### Week 2
- 

### Week 3
-