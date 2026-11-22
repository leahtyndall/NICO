"""
Live microphone transcription with MediaPipe AudioTranscriber (Parakeet TDT).

Install:
    pip install mediapipe sounddevice numpy

Run:
    python live_transcribe.py
    Ctrl+C to stop.
"""

import queue
import sys

import numpy as np
import sounddevice as sd
import mediapipe as mp

from pathlib import Path
MODEL_PATH = str(Path(__file__).parent / "model" / "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm")
SAMPLE_RATE = 16000
CHUNK_SEC = 0.5          # how much audio to feed per transcribe() call
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)
DEVICE = 2            # None = default mic. Run `python -m sounddevice` to list devices.

BaseOptions = mp.tasks.BaseOptions
AudioTranscriber = mp.tasks.audio.AudioTranscriber
AudioTranscriberOptions = mp.tasks.audio.AudioTranscriberOptions

audio_q: "queue.Queue[np.ndarray]" = queue.Queue()


def mic_callback(indata, frames, time_info, status):
    if status:
        print(f"\n[mic] {status}", file=sys.stderr)
    # indata is (frames, channels) float32 in [-1, 1]; take mono channel
    audio_q.put(indata[:, 0].copy())


def main():
    options = AudioTranscriberOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU,
        ),
        sample_rate_hz=SAMPLE_RATE,
        enable_interim_results=True,
    )

    with AudioTranscriber.create_from_options(options) as transcriber, \
         sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="float32",
                        blocksize=CHUNK_SAMPLES,
                        device=DEVICE,
                        callback=mic_callback):
        print("Listening... (Ctrl+C to stop)\n")
        last_text = ""
        try:
            while True:
                chunk = audio_q.get()  # blocks until the mic delivers a chunk
                result = transcriber.transcribe(chunk, end_of_stream=False)
                text = (result.text or "").strip()
                if text and text != last_text:
                    # overwrite the current line with the latest interim text
                    print(f"\r\033[K{text}", end="", flush=True)
                    last_text = text
        except KeyboardInterrupt:
            pass

        # flush whatever's still queued, then close the stream
        while not audio_q.empty():
            transcriber.transcribe(audio_q.get(), end_of_stream=False)
        final = transcriber.finish_stream()
        print(f"\n\nFinal: {final.text}")


if __name__ == "__main__":
    main()