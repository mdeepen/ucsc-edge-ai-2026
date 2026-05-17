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
| 04 | Custom CNN + NPU Deployment | ✅ | Garbage classifier int8 → Vela → AE3 Ethos-U55, 100% NPU |

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

## Week 04 — Custom CNN + NPU Deployment
- Designed and trained a custom garbage classification CNN in Google Colab
- 6 classes: cardboard, glass, metal, paper, plastic, trash
- Input: 96×96 RGB, architecture: Conv-BN-ReLU blocks → GlobalAvgPool → Dense
- Quantized to int8 using TFLite full-integer quantization with representative dataset
- Compiled for AE3 Ethos-U55 NPU using Arm Vela compiler
- Diagnosed and fixed `Invoke failed` error caused by wrong Vela system config (`Ethos_U55_High_End_Embedded` → `RTSS_HP_SRAM_OSPI` from AE3-specific `vela.ini`)
- All 39 operators offloaded to NPU (0% CPU fallback)
- Deployed and ran live inference on OpenMV AE3 via `classify_garbage_openmv.py`
- Installed permanent `vela-ae3` wrapper script for future AE3 model compilation

**Files:** `week04-Models/assignment/`
- `GarbageClassification_CNN.ipynb` — training notebook (Colab)
- `classify_garbage_openmv.py` — OpenMV inference script (96×96 windowed camera)
- `models/` — int8 and Vela-compiled `.tflite` models + labels
- `results/` — models, inference screenshots, Vela compilation notes

# Setup
- Host: Windows 11 + WSL2 Ubuntu
- IDE: OpenMV IDE + VS Code (WSL)
- See `setup/wsl2_setup.md` for full environment setup

# Hardware
OpenMV AE3 : Arm Cortex-M55 CPU + Ethos-U55 NPU
