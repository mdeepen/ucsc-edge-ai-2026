# Micro Speech inference on OpenMV AE3
#
# Model: tiny_conv trained on "yes" / "no" keywords
# Input: 1960 int8 values = 49 frames x 40 mel bins (1 second @ 16 kHz)
# Preprocessing matches TF micro speech "micro" mode:
#   - 30 ms Hann-windowed frames, 20 ms stride
#   - 512-point FFT -> mel filterbank (40 bins, 20–7600 Hz) -> log
#   - Quantize float -> int8 using model's scale / zero_point
#
# Sliding window: 50% overlap — advances 0.5 s per step, so words at any
# boundary are captured in the next window.
#
# Copy to board:
#   microspeech_model_vela.tflite   (Vela-compiled int8 model)
#   microspeech_labels.txt          (yes / no / _silence_ / _unknown_)

import audio
import ml
import gc
import utime
import math
import struct
from ulab import numpy as np   # correct MicroPython import (not "import ulab.numpy")

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
WINDOW_SIZE   = 480    # 30 ms
WINDOW_STRIDE = 320    # 20 ms
FFT_SIZE      = 512
N_MELS        = 40
N_FRAMES      = 49
FMIN          = 20.0
FMAX          = 7600.0

INPUT_SCALE      = 0.10171568   # from model input quantization
INPUT_ZERO_POINT = -128

YES_THRESHOLD    = 0.30   # low bar: yes gets 30-97% depending on window alignment
NO_THRESHOLD     = 0.70   # higher bar: suppresses false "no" from "yes" onset
ENERGY_THRESHOLD = 2500   # PCM max below this → silence

# ── Mel filterbank: (N_MELS, FFT_SIZE//2+1) ───────────────────────────────────
def _hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def build_mel_filterbank():
    n_fft_bins    = FFT_SIZE // 2 + 1
    freq_per_bin  = SAMPLE_RATE / FFT_SIZE
    mel_min       = _hz_to_mel(FMIN)
    mel_max       = _hz_to_mel(FMAX)
    mel_pts = [mel_min + i * (mel_max - mel_min) / (N_MELS + 1)
               for i in range(N_MELS + 2)]
    bin_pts = [min(int(_mel_to_hz(m) / freq_per_bin), n_fft_bins - 1)
               for m in mel_pts]

    fb = np.zeros((N_MELS, n_fft_bins), dtype=np.float)
    for m in range(1, N_MELS + 1):
        lo, ctr, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo, ctr):
            if ctr != lo:
                fb[m - 1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi != ctr:
                fb[m - 1, k] = (hi - k) / (hi - ctr)
    return fb

# ── Hann window ───────────────────────────────────────────────────────────────
def build_hann_window(size):
    return np.array(
        [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (size - 1)) for i in range(size)],
        dtype=np.float)

# ── Log-mel feature extraction ────────────────────────────────────────────────
# ulab fft.fft(real_array) returns a (re, im) tuple — no complex dtype needed.
def compute_features(pcm, mel_fb, hann_win):
    n_fft_bins = FFT_SIZE // 2 + 1
    features   = np.zeros((N_FRAMES, N_MELS), dtype=np.float)
    padded     = np.zeros(FFT_SIZE, dtype=np.float)

    for frame_idx in range(N_FRAMES):
        start = frame_idx * WINDOW_STRIDE
        padded[:WINDOW_SIZE] = pcm[start: start + WINDOW_SIZE] * hann_win
        padded[WINDOW_SIZE:] = 0.0

        # ulab returns (real_part, imag_part) tuple for real-valued input
        re, im = np.fft.fft(padded)

        # One-sided power spectrum (DC through Nyquist)
        power = re[:n_fft_bins] * re[:n_fft_bins] + im[:n_fft_bins] * im[:n_fft_bins]

        # Mel filterbank via matrix-vector dot product, then log
        mel_energy      = np.dot(mel_fb, power)        # (N_MELS,)
        features[frame_idx] = np.log(mel_energy + 1e-6)

    return features


# ── Audio recording — sliding window DMA ──────────────────────────────────────
# Chunk size confirmed: 1024 bytes = 512 samples per callback.
#
# Layout: pcm_raw holds N_CHUNKS chunks (≈ 1 s).
# After each inference we keep the newest KEEP_CHUNKS and record N_HALF_CHUNKS
# fresh chunks — so each step advances ~0.5 s and words are never cut off.
#
#   First step : record all 31 chunks → infer
#   Every later: shift buffer left by N_HALF_CHUNKS, record 15 new chunks → infer
#
CHUNK_SAMPLES  = 512
N_CHUNKS       = 31                              # total window: 31 × 512 = 15872 samples ≈ 1 s
N_HALF_CHUNKS  = N_CHUNKS // 2                  # 15 — new chunks recorded each step
KEEP_CHUNKS    = N_CHUNKS - N_HALF_CHUNKS        # 16 — chunks retained from previous window
TOTAL_SAMPLES  = N_CHUNKS * CHUNK_SAMPLES        # 15872
KEEP_BYTES     = KEEP_CHUNKS * CHUNK_SAMPLES * 2 # 16384 — bytes retained
NEW_BYTES      = N_HALF_CHUNKS * CHUNK_SAMPLES * 2  # 15360 — bytes of fresh audio

pcm_raw    = bytearray(TOTAL_SAMPLES * 2)  # 31744 bytes
chunk_idx  = [0]
_first_run = [True]

def audio_callback(buf):
    if _first_run[0]:
        # Warm-up: fill entire buffer
        if chunk_idx[0] < N_CHUNKS:
            off = chunk_idx[0] * CHUNK_SAMPLES * 2
            pcm_raw[off: off + CHUNK_SAMPLES * 2] = buf
            chunk_idx[0] += 1
    else:
        # Sliding: fill only the second half (offset past the retained region)
        if chunk_idx[0] < N_HALF_CHUNKS:
            off = KEEP_BYTES + chunk_idx[0] * CHUNK_SAMPLES * 2
            pcm_raw[off: off + CHUNK_SAMPLES * 2] = buf
            chunk_idx[0] += 1

# ── Startup ───────────────────────────────────────────────────────────────────
print("Loading model...")
gc.collect()
try:
    model = ml.Model("microspeech_model_vela.tflite", load_to_fb=True)
except Exception as e:
    raise Exception("Copy microspeech_model_vela.tflite to board. (" + str(e) + ")")

try:
    labels = [line.rstrip('\n') for line in open("microspeech_labels.txt")]
    # Notebook wrote [yes, no, silence, unknown] but TF model outputs
    # in prepare_words_list order: [silence, unknown, yes, no]
    labels = labels[2:] + labels[:2]
except Exception as e:
    raise Exception("Copy microspeech_labels.txt to board. (" + str(e) + ")")

YES_IDX = labels.index('yes')
NO_IDX  = labels.index('no')

print("Input shape :", model.input_shape)
print("Labels      :", labels)

print("Building mel filterbank (one-time)...")
mel_fb   = build_mel_filterbank()
hann_win = build_hann_window(WINDOW_SIZE)
print("Ready — listening (sliding window, ~0.5 s step)...")

audio.init(channels=1, frequency=SAMPLE_RATE, gain_db=24)


# ── Inference loop ────────────────────────────────────────────────────────────
# Burst-print state: a word typically spans 2 consecutive windows due to 50%
# overlap.  We buffer the most recent keyword and only print it once the burst
# ends (energy drops below threshold).  The last detection in a burst is the
# most completely-captured occurrence of the word — more reliable than the
# first (partial) capture, which often misclassifies.
_pending_lbl = [None]
_pending_p   = [0.0]
_kw_steps    = [0]    # consecutive steps with a keyword above threshold

while True:
    # --- record audio ---
    chunk_idx[0] = 0
    audio.start_streaming(audio_callback)
    if _first_run[0]:
        while chunk_idx[0] < N_CHUNKS:
            utime.sleep_ms(10)
    else:
        while chunk_idx[0] < N_HALF_CHUNKS:
            utime.sleep_ms(10)
    audio.stop_streaming()
    _first_run[0] = False

    # Decode BEFORE sliding: ints is a snapshot of the correct 1-second window.
    # Window layout: pcm_raw[0:KEEP_BYTES] = retained old audio,
    #                pcm_raw[KEEP_BYTES:]   = fresh audio just recorded.
    ints = struct.unpack('<%dh' % TOTAL_SAMPLES, pcm_raw)

    # Slide now (safe: ints already decoded from pcm_raw).
    pcm_raw[:KEEP_BYTES] = pcm_raw[NEW_BYTES: NEW_BYTES + KEEP_BYTES]

    # Energy gate — check the FULL window.
    # Checking only the new half caused missed detections when a word landed
    # entirely in the retained (old) half of the buffer.
    pcm_max = max(abs(ints[i]) for i in range(0, TOTAL_SAMPLES, 4))
    if pcm_max < ENERGY_THRESHOLD:
        if _pending_lbl[0] is not None:
            # Burst just ended — print the last (most complete) detection.
            print(">>> %-12s  %.0f%%" % (_pending_lbl[0], _pending_p[0] * 100))
            _pending_lbl[0] = None
            _pending_p[0]   = 0.0
            _kw_steps[0]    = 0
        else:
            print("    _silence_")
        gc.collect()
        continue

    pcm_f32 = np.array(ints, dtype=np.float)

    # pad to exactly SAMPLE_RATE samples
    pcm_1s = np.zeros(SAMPLE_RATE, dtype=np.float)
    pcm_1s[:TOTAL_SAMPLES] = pcm_f32

    # --- features -> inference ---
    features = compute_features(pcm_1s, mel_fb, hann_win)

    # Manually quantize: ml.Model casts float→int8 directly, no scale/zp applied.
    flat       = features.flatten()
    q          = flat * (1.0 / INPUT_SCALE) + INPUT_ZERO_POINT
    q          = np.clip(q, -128, 127)
    feat_input = q.reshape((1, N_FRAMES * N_MELS))

    output = model.predict([feat_input])[0]

    # Model graph includes softmax: output is already a probability distribution.
    probs = output.flatten().tolist()
    yes_p = probs[YES_IDX]
    no_p  = probs[NO_IDX]

    # Debug: show yes/no scores on every non-silent inference.
    print("dbg y%d n%d u%d" % (yes_p * 100, no_p * 100, probs[1] * 100))

    # Asymmetric thresholds: yes gets a lower bar because Indian accent produces
    # lower yes-confidence scores.  Check yes first so a window where both
    # classes are elevated preferrs yes.  no requires 75% to suppress the false
    # "no" that fires on the /j/ onset of "yes".
    if yes_p >= YES_THRESHOLD:
        kw_lbl, kw_p = labels[YES_IDX], yes_p
    elif no_p >= NO_THRESHOLD:
        kw_lbl, kw_p = labels[NO_IDX], no_p
    else:
        kw_lbl, kw_p = None, 0.0

    if kw_lbl is not None:
        # Yes-sticky: once "yes" is buffered, a later "no" window in the same
        # burst cannot overwrite it.  The sliding window means "yes" can appear
        # at 97% confidence in one frame and trigger "no" at 81% in the next —
        # the first clean "yes" must survive.
        # "no" → "yes" transition is still allowed (burst 9 pattern).
        if not (_pending_lbl[0] == labels[YES_IDX] and kw_lbl == labels[NO_IDX]):
            _pending_lbl[0] = kw_lbl
            _pending_p[0]   = kw_p
        _kw_steps[0]   += 1
        if _kw_steps[0] >= 4:
            # Safety valve: flush if keyword runs unbroken for > 4 steps (~2 s)
            print(">>> %-12s  %.0f%%" % (_pending_lbl[0], _pending_p[0] * 100))
            _pending_lbl[0] = None
            _pending_p[0]   = 0.0
            _kw_steps[0]    = 0
    else:
        # Neither keyword detected — burst is over.
        if _pending_lbl[0] is not None:
            print(">>> %-12s  %.0f%%" % (_pending_lbl[0], _pending_p[0] * 100))
            _pending_lbl[0] = None
            _pending_p[0]   = 0.0
        _kw_steps[0] = 0

    gc.collect()
