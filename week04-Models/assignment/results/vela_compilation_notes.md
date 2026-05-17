# Compiling int8 TFLite Model for OpenMV AE3 using Vela

## What is Vela?

Vela is Arm's open-source compiler for optimizing TFLite models to run on the
Ethos-U Neural Processing Unit (NPU). It takes a standard int8 quantized
`.tflite` model and produces a new `.tflite` file with NPU-accelerated
operator scheduling and memory layout optimized for the target hardware.

Without Vela, the model runs on the Cortex-M55 CPU only.
With Vela, all 39 operators run on the Ethos-U55 NPU — significantly faster
and more power-efficient.

## OpenMV AE3 Hardware

- SoC: Alif Ensemble E3 (AE302F80F55D5AE)
- CPU: Arm Cortex-M55 HP core @ 400 MHz
- NPU: Arm Ethos-U55 — 256 MACs
- Memory: SRAM + OSPI Flash

## Why Generic Vela Settings Don't Work

The Vela compiler ships with generic system configs (`Ethos_U55_High_End_Embedded`,
etc.) that describe hypothetical reference hardware. The AE3 uses a custom
memory layout specific to the Alif Ensemble chip.

Using the generic config causes `ValueError: Invoke failed` at runtime because
the compiled memory addresses and bandwidth assumptions do not match the
actual hardware.

The OpenMV IDE ships with an AE3-specific `vela.ini` at:
```
/applications/openmv-ide-install/share/qtcreator/firmware/OPENMV_AE3/vela.ini
```

The correct settings for the AE3 HP core are read from `settings.json` in the
OpenMV IDE firmware folder:
- `--accelerator-config ethos-u55-256`
- `--system-config RTSS_HP_SRAM_OSPI`
- `--memory-mode Shared_Sram`

## Installation

```bash
pip install ethos-u-vela
```

## Compile Command

```bash
vela custom_objects_int8.tflite \
  --config /applications/openmv-ide-install/share/qtcreator/firmware/OPENMV_AE3/vela.ini \
  --system-config RTSS_HP_SRAM_OSPI \
  --accelerator-config ethos-u55-256 \
  --memory-mode Shared_Sram \
  --output-dir ./results/
```

Output file: `custom_objects_int8_vela.tflite`

## Compilation Summary

| Parameter             | Value                  |
|-----------------------|------------------------|
| Accelerator           | Ethos_U55_256          |
| System config         | RTSS_HP_SRAM_OSPI      |
| Memory mode           | Shared_Sram            |
| Core clock            | 400 MHz                |
| Total SRAM used       | 74.78 KiB              |
| Total Flash used      | 44.27 KiB              |
| CPU operators         | 0 (0%)                 |
| NPU operators         | 39 (100%)              |
| Neural network MACs   | 16,991,884 MACs/batch  |

All 39 operators are offloaded to the Ethos-U55 NPU — zero CPU fallback.

## Permanent Wrapper Script

A reusable script `vela-ae3` was installed at `~/.local/bin/vela-ae3` so any
future model can be compiled with:

```bash
vela-ae3 my_model_int8.tflite
# output: my_model_int8_vela.tflite (same directory)
```
