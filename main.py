import enum
import time
from statistics import mean
import os
import concurrent.futures
import shutil
from pathlib import Path
from openai import OpenAI

OPENAI_SPEACH_MODEL = "onyx"
CHARACTER_LIMIT = 4000
OUTPUT_DIR = "Output"

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

print("Preparing to generate audio")

# Put through gpt
client = OpenAI()

audio_responces = {}
count_finished = 0
responce_times: list[float] = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    for i, chunk in enumerate(chunks):

        def get_responce(i: int, chunk: str):
            responce_time_start = time.time()

            response = client.audio.speech.create(
                model="tts-1-hd", voice=OPENAI_SPEACH_MODEL, input=chunk
            )
            audio_responces[i] = response

            responce_time_end = time.time()
            time_taken = responce_time_end - responce_time_start
            global response_times
            responce_times.append(time_taken)

            global count_finished
            count_finished += 1
            print(f"Chunk finished, took {time_taken:.2f} seconds")
            print(f"{int(count_finished / len(chunks) * 100)}% Generated")

        executor.submit(get_responce, i, chunk)

print("Generation finished, preparing to write to disk")

print("Sorting audio responces")
sorted_audio_responces = [value for _, value in sorted(audio_responces.items())]

# Write to file
for i, response in enumerate(sorted_audio_responces):
    speech_file_path = Path(__file__).parent / OUTPUT_DIR / f"speech - {i}.mp3"
    print(f"Writing {speech_file_path}")
    response.stream_to_file(speech_file_path)


time_taken = time.time() - start_time
print(
    f"Finished successfully, took {time_taken:.2f} seconds, average generation time was {mean(responce_times):.2f}"
)
