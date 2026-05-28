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
| 05 | Keyword Spotting on Edge | ✅ | "yes/no" tiny_conv on AE3 NPU — 87.6 µs inference, sliding-window DMA capture |

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

## Week 05 — Keyword Spotting (RNN / Micro Speech) on Edge
- Downloaded pretrained TF Micro Speech `tiny_conv` model (18 KB int8, "yes" / "no" keywords)
- Implemented log-mel spectrogram preprocessing in MicroPython ulab: 49 frames × 40 bins, 30 ms Hann-windowed frames, 20 ms stride, 512-pt FFT, mel filterbank 20–7600 Hz
- Applied manual int8 quantization (scale = 0.10171568, zero_point = −128) — `ml.Model.predict()` on AE3 does not apply the model's quantization parameters automatically
- Compiled with Arm Vela for AE3 Ethos-U55 NPU: **87.6 µs inference**, 11,414 inf/s, all 29 ops on NPU (0% CPU fallback)
- Deployed real-time keyword spotting on OpenMV AE3 via callback-based DMA audio capture
- Implemented **50% overlap sliding window** (0.5 s step, 1 s window) so words spoken at any time are captured — avoids missed detections at non-overlapping boundaries
- Diagnosed label-order mismatch between notebook output (`[yes, no, silence, unknown]`) and TF `prepare_words_list` order (`[silence, unknown, yes, no]`)
- Added diagnostic probability output to identify model generalization gap for Indian accent — model gives 30–97% yes-confidence depending on window alignment vs. 70–93% false no-confidence on word onset
- Implemented **asymmetric yes/no thresholds** (yes ≥ 30%, no ≥ 70%) and **yes-sticky burst logic** (once "yes" is buffered in a burst, a subsequent "no" window cannot overwrite it) to recover correct detections
- Built `ae3_recorder.py` MicroPython script for on-device audio dataset collection (saves WAV files to SD card)

**Files:** `week05-RNN-For-EdgeAI/`
- `Train_micro_speech_model_only.ipynb` — notebook: download checkpoint, quantize, export TFLite
- `models/` — float32, int8, and C-array versions of the tiny_conv model
- `results/microspeech_openmv.py` — OpenMV AE3 inference script (sliding window, burst-print, asymmetric thresholds)
- `results/microspeech_model_vela.tflite` — Vela-compiled model for Ethos-U55
- `results/microspeech_model_summary_RTSS_HP_SRAM_OSPI.csv` — Vela performance report
- `owndataset/ae3_recorder.py` — on-device WAV recorder for collecting custom keyword samples
- `owndataset/collect_audio.py` — host-side script to trigger recordings over USB serial

# Setup
- Host: Windows 11 + WSL2 Ubuntu
- IDE: OpenMV IDE + VS Code (WSL)
- See `setup/wsl2_setup.md` for full environment setup

# Hardware
OpenMV AE3 : Arm Cortex-M55 CPU + Ethos-U55 NPU
