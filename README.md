# ucsc-edge-ai-2026
Edge AI assignments for UCSC Extension (AI-Driven Embedded Systems) 2026.
Hardware: OpenMV AE3 (Cortex-M55 + Ethos-U55 NPU).
Covers Machine Learning, TFLite inference, Edge Impulse model training, keyword spotting, and on-device computer vision.
Perspective: Exploring Edge AI concepts for embedded systems engineering.

# Weekly Assignments

| Week | Topic | Status | Key Result |
|------|-------|--------|------------|
| 01 | OpenMV Setup + Built-in AI Examples | ✅ | FOMO face detection @ 60fps |
| 02 | Neural Network from Scratch | ✅ | Checkerboard classifier k=2 and k=4 (NumPy only) |

## Week 01 — OpenMV Setup + Built-in AI Examples
- Connected OpenMV AE3 to OpenMV IDE
- Ran built-in object detection and speech examples on-device
- Trained FOMO model on Edge Impulse (lego bricks vs wooden blocks, 2 classes)
- Ran Edge Impulse TFLite model on AE3 via `ei_object_detection.py`

**Files:** `week01-openmv/`
- `edge-impulse-model/` — trained `.tflite` + labels + inference script
- `edge-impulse-dataset/` — captured dataset (lego bricks, wooden blocks)
- `openmvide-ae3-examples/` — screenshots of built-in AI demos

## Week 02 — Neural Network from Scratch
- Built a 2-layer neural network using NumPy only (no ML libraries)
- Implemented: weight init, forward pass, sigmoid activation, BCE loss, backpropagation, gradient descent
- Trained on a 2D checkerboard classification problem
- Visualized learned decision boundary
- Extended to a deep 4-layer network [2→64→64→64→64→1] with ReLU, He init, mini-batch SGD, and early stopping

**Files:** `week02-NN-From-Scratch/`
- `assignment/checkerboard_nn_assignment_k=2.ipynb` — 2-class checkerboard (2-layer network)
- `assignment/checkerboard_nn_assignment_k=4.ipynb` — 4-class checkerboard (deep network, ReLU, mini-batch SGD, early stopping)
- `notebooks/` — supplementary OpenMV + desktop prototyping notebooks

# Setup
- Host: Windows 11 + WSL2 Ubuntu
- IDE: OpenMV IDE + VS Code (WSL)
- See `setup/wsl2_setup.md` for full environment setup

# Hardware
OpenMV AE3 : Arm Cortex-M55 CPU + Ethos-U55 NPU
