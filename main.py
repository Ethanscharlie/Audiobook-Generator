import enum
import time
import subprocess
from statistics import mean
import os
import concurrent.futures
import shutil
from pathlib import Path

CHARACTER_LIMIT = 10_000
OUTPUT_DIR = "Output"
MODEL_FILE = "piper/en_US-norman-medium.onnx"
MODEL_CONFIG_FILE = "piper/en_en_US_norman_medium_en_US-norman-medium.onnx.json"

start_time = time.time()

# Wipe downloads dir
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
else:
    shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)

print("Reading from file text.txt")

# Grab the words from the book
text_from_file = ""
with open("text.txt") as f:
    text_from_file = f.read()

print(f"Spliting text into chunks of size {CHARACTER_LIMIT} characters")

# Split into chunks
chunks: list[str] = [""]
counter = 0
for c in text_from_file:
    counter += 1
    if counter >= CHARACTER_LIMIT and c == " ":
        chunks.append("")
        counter = 0
        continue

    chunks[-1] += c

print(f"Made {len(chunks)} chunks")
print("Preparing to generate audio")

# Put through piper
count_finished = 0
responce_times: list[float] = []
for i, chunk in enumerate(chunks):
    responce_time_start = time.time()

    subprocess.run(
        f"piper-tts --model {MODEL_FILE} --config {MODEL_CONFIG_FILE} -q --output_file {OUTPUT_DIR}/SPEECH-{i}.wav",
        input=chunk,
        text=True,
        shell=True,
    )

    responce_time_end = time.time()
    time_taken = responce_time_end - responce_time_start
    responce_times.append(time_taken)

    count_finished += 1
    print(f"Chunk finished, took {time_taken:.2f} seconds")
    print(f"{int(count_finished / len(chunks) * 100)}% Generated ({count_finished} / {len(chunks)})")
    print(f"ETA: {(len(chunks) - count_finished) * mean(responce_times) / 60:.2f} minutes")

time_taken = time.time() - start_time
print(
    f"Finished successfully, took {time_taken:.2f} seconds, average generation time was {mean(responce_times):.2f}"
)
