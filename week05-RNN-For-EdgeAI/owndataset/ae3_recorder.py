import audio, os, time

SAMPLE_RATE  = 16000
NUM_SAMPLES  = 16000   # 1 second
WORD         = "up"    # ← change this per label
DELAY_BETWEEN = 3000   # ms between recordings — gives you time to speak

import os
print(os.listdir('/'))

# Setup folder
path = f"/sd/dataset/{WORD}"
try:
    os.makedirs(path)
except:
    pass

def next_index():
    return len([f for f in os.listdir(path) if f.endswith(".wav")])

buf = bytearray(NUM_SAMPLES * 2)
audio.init(channels=1, frequency=SAMPLE_RATE, gain_db=24, highpass=0.9883)

print(f"Recording '{WORD}' every {DELAY_BETWEEN//1000}s. Say the word when you see 'Recording...'")
print("Press Ctrl+C to stop.\n")

while True:
    idx = next_index()
    print(f"[{idx}] Get ready...")
    time.sleep_ms(DELAY_BETWEEN - 1000)   # countdown
    print(f"[{idx}] Recording... 🔴 SAY IT NOW")
    audio.start_streaming(buf)
    time.sleep_ms(1100)
    audio.stop_streaming()

    fname = f"{path}/{WORD}_{idx:04d}.wav"
    with open(fname, "wb") as f:
        n = len(buf)
        f.write(b'RIFF')
        f.write((36 + n).to_bytes(4, 'little'))
        f.write(b'WAVEfmt ')
        f.write((16).to_bytes(4, 'little'))
        f.write((1).to_bytes(2, 'little'))
        f.write((1).to_bytes(2, 'little'))
        f.write((16000).to_bytes(4, 'little'))
        f.write((32000).to_bytes(4, 'little'))
        f.write((2).to_bytes(2, 'little'))
        f.write((16).to_bytes(2, 'little'))
        f.write(b'data')
        f.write(n.to_bytes(4, 'little'))
        f.wri