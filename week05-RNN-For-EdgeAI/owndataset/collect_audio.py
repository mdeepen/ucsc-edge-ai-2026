"""
collect_audio.py — Run this on your PC
=======================================
Collects labelled 16kHz mono WAV clips from the AE3 board over USB-serial
and organises them into the directory structure expected by train.py.

Usage
-----
  python collect_audio.py --word up --count 100
  python collect_audio.py --word _background_noise_ --count 1 --duration 30
  python collect_audio.py --scan          # auto-detect COM port and exit

Requirements
------------
  pip install pyserial

Dataset layout produced
-----------------------
  dataset/
  ├── up/           ← one .wav per recording
  ├── down/
  ├── left/
  ├── right/
  ├── on/
  ├── off/
  └── _background_noise_/   ← long ambient noise files
"""

import argparse
import os
import struct
import sys
import time
import wave
import glob

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("[ERROR] pyserial not found.  Run:  pip install pyserial")

# ── Constants — must match ae3_recorder.py ────────────────────────────────────
SAMPLE_RATE    = 16000
CLIP_DURATION  = 1.0
NUM_SAMPLES    = int(SAMPLE_RATE * CLIP_DURATION)
BYTES_PER_SAMP = 2
CLIP_BYTES     = NUM_SAMPLES * BYTES_PER_SAMP

BAUD_RATE      = 115200
READ_TIMEOUT   = 5       # seconds to wait for board response
DATASET_DIR    = "dataset"

# ── Colour helpers (ANSI) ─────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):    print(f"{GREEN}  ✓  {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}  ⚠  {msg}{RESET}")
def err(msg):   print(f"{RED}  ✗  {msg}{RESET}")
def info(msg):  print(f"{CYAN}     {msg}{RESET}")

# ── Port detection ────────────────────────────────────────────────────────────
def find_ae3_port() -> str | None:
    """
    Return the first serial port that looks like an OpenMV / STM32 VCP,
    or None if nothing found.
    """
    candidates = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        mfr  = (p.manufacturer or "").lower()
        if any(k in desc or k in mfr for k in
               ("openmv", "stm32", "stlink", "virtual com", "cdc")):
            candidates.append(p.device)

    if candidates:
        return candidates[0]

    # Fallback: return the first non-Bluetooth port
    all_ports = [p.device for p in serial.tools.list_ports.comports()
                 if "bluetooth" not in (p.description or "").lower()]
    return all_ports[0] if all_ports else None


def ping_board(ser: serial.Serial) -> bool:
    """Send a ping and wait for OK response."""
    ser.reset_input_buffer()
    ser.write(b'p')
    deadline = time.time() + 3
    resp = b''
    while time.time() < deadline:
        resp += ser.read(ser.in_waiting or 1)
        if b'OK' in resp:
            return True
    return False


# ── WAV helpers ───────────────────────────────────────────────────────────────
def save_wav(path: str, raw_pcm: bytes):
    """Write raw int16-LE PCM bytes to a mono 16kHz WAV file."""
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(BYTES_PER_SAMP)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_pcm)


def verify_wav(path: str) -> bool:
    """Quick sanity-check on a saved WAV."""
    try:
        with wave.open(path, 'rb') as wf:
            return (wf.getnchannels()  == 1 and
                    wf.getsampwidth()  == BYTES_PER_SAMP and
                    wf.getframerate()  == SAMPLE_RATE and
                    wf.getnframes()    >= NUM_SAMPLES - 100)
    except Exception:
        return False


def rms_level(raw_pcm: bytes) -> float:
    """Return RMS amplitude (0–32768) of the clip."""
    samples = struct.unpack(f"<{NUM_SAMPLES}h", raw_pcm[:CLIP_BYTES])
    mean_sq = sum(s * s for s in samples) / NUM_SAMPLES
    return mean_sq ** 0.5


# ── Recording ─────────────────────────────────────────────────────────────────
def record_one(ser: serial.Serial) -> bytes | None:
    """
    Trigger one recording on the board.
    Returns CLIP_BYTES of raw PCM, or None on failure.
    """
    ser.reset_input_buffer()
    ser.write(b'r')

    # Read exactly CLIP_BYTES + 4 (for 'DONE' or 'ERR!')
    data = b''
    deadline = time.time() + READ_TIMEOUT
    while len(data) < CLIP_BYTES + 4 and time.time() < deadline:
        chunk = ser.read(min(4096, CLIP_BYTES + 4 - len(data)))
        data += chunk

    if len(data) < CLIP_BYTES + 4:
        err(f"Timeout: only got {len(data)} bytes (expected {CLIP_BYTES + 4})")
        return None

    pcm  = data[:CLIP_BYTES]
    tail = data[CLIP_BYTES:CLIP_BYTES + 4]

    if tail == b'ERR!':
        err("Board reported a recording error.")
        return None
    if tail != b'DONE':
        warn(f"Unexpected trailer: {tail!r} — proceeding anyway.")

    return pcm


# ── Next available filename ───────────────────────────────────────────────────
def next_filename(word_dir: str, word: str) -> str:
    """Return the next non-colliding filename like up_0042.wav"""
    existing = glob.glob(os.path.join(word_dir, f"{word}_*.wav"))
    indices  = []
    for f in existing:
        base = os.path.splitext(os.path.basename(f))[0]
        try:
            indices.append(int(base.split('_')[-1]))
        except ValueError:
            pass
    nxt = max(indices, default=-1) + 1
    return os.path.join(word_dir, f"{word}_{nxt:04d}.wav")


# ── Dataset stats ─────────────────────────────────────────────────────────────
def print_dataset_stats():
    print(f"\n{CYAN}── Dataset summary ({'─' * 40}){RESET}")
    if not os.path.isdir(DATASET_DIR):
        warn("No dataset directory found yet.")
        return
    for label in sorted(os.listdir(DATASET_DIR)):
        d = os.path.join(DATASET_DIR, label)
        if os.path.isdir(d):
            count = len(glob.glob(os.path.join(d, "*.wav")))
            bar   = "█" * min(count // 5, 40)
            colour = GREEN if count >= 150 else (YELLOW if count >= 50 else RED)
            print(f"  {label:<30s} {colour}{count:>4d} clips  {bar}{RESET}")
    print()


# ── Main entry point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Collect audio from AE3 board for micro_speech training")
    parser.add_argument("--word",  "-w", default="up",
                        help="Label / word to record (e.g. up, down, left)")
    parser.add_argument("--count", "-n", type=int, default=50,
                        help="Number of clips to collect (default: 50)")
    parser.add_argument("--port",  "-p", default=None,
                        help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud",  "-b", type=int, default=BAUD_RATE)
    parser.add_argument("--min-rms", type=float, default=200.0,
                        help="Reject clips quieter than this RMS (default 200)")
    parser.add_argument("--scan", action="store_true",
                        help="List available serial ports and exit")
    parser.add_argument("--stats", action="store_true",
                        help="Print dataset statistics and exit")
    args = parser.parse_args()

    if args.stats:
        print_dataset_stats()
        return

    if args.scan:
        print("\nAvailable serial ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device:<20s}  {p.description}")
        return

    # ── Resolve port ──────────────────────────────────────────────────────────
    port = args.port or find_ae3_port()
    if not port:
        err("No serial port found. Use --port /dev/ttyACM0 (or COMx on Windows).")
        sys.exit(1)

    # ── Create output directory ───────────────────────────────────────────────
    word_dir = os.path.join(DATASET_DIR, args.word)
    os.makedirs(word_dir, exist_ok=True)

    # ── Connect ───────────────────────────────────────────────────────────────
    info(f"Connecting to {port} @ {args.baud} baud…")
    try:
        ser = serial.Serial(port, args.baud, timeout=READ_TIMEOUT)
    except serial.SerialException as e:
        err(f"Cannot open port: {e}")
        sys.exit(1)

    time.sleep(1.5)   # let board reset if it reboots on DTR
    if not ping_board(ser):
        err("Board did not respond to ping. Is ae3_recorder.py running?")
        ser.close()
        sys.exit(1)
    ok(f"Board is alive on {port}")

    # ── Collection loop ───────────────────────────────────────────────────────
    collected = 0
    skipped   = 0
    target    = args.count

    print(f"\n{CYAN}Recording {target} clips for label '{args.word}'{RESET}")
    print(f"  Output → {os.path.abspath(word_dir)}")
    print(f"  Min RMS threshold: {args.min_rms:.0f}\n")

    while collected < target:
        remaining = target - collected
        prompt = (f"[{collected+1}/{target}]  "
                  f"Press Enter to record '{args.word}' "
                  f"(q = quit, s = show stats): ")
        try:
            user = input(prompt).strip().lower()
        except KeyboardInterrupt:
            print()
            break

        if user == 'q':
            break
        if user == 's':
            print_dataset_stats()
            continue

        info("Recording…  🔴")
        pcm = record_one(ser)
        if pcm is None:
            skipped += 1
            warn("Recording failed — skipping.")
            continue

        rms = rms_level(pcm)
        if rms < args.min_rms:
            skipped += 1
            warn(f"Clip too quiet (RMS {rms:.0f} < {args.min_rms:.0f}) — discarded. "
                 f"Speak louder or adjust --min-rms.")
            continue

        path = next_filename(word_dir, args.word)
        save_wav(path, pcm)

        if not verify_wav(path):
            skipped += 1
            warn(f"WAV verification failed: {path}")
            os.remove(path)
            continue

        collected += 1
        ok(f"Saved: {os.path.basename(path)}  (RMS {rms:.0f})")

    ser.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{GREEN}Done.{RESET}  Collected {collected} clips, skipped {skipped}.")
    print_dataset_stats()

    # ── Notebook integration hint ─────────────────────────────────────────────
    print(f"{CYAN}Next steps:{RESET}")
    print(f"  1. Collect ≥150 clips per word.")
    print(f"  2. Collect silence:  python collect_audio.py --word _background_noise_ --count 20")
    print(f"  3. In the notebook, set:")
    print(f"       DATA_URL   = ''")
    print(f"       DATASET_DIR = 'dataset/'")
    print(f"     and choose 'train' when prompted.\n")


if __name__ == "__main__":
    main()
