"""
Look for <voice-name>.wav files in CUSTOM_VOICE_DIR and convert them to .npy
using any corresponding <voice-name>.txt as prompt text if it exists;
      otherwise fallback to CACHED_PROMPT_TEXT
"""
import glob
import os

VOICE_CACHE_DIR = "./npy"
CUSTOM_VOICE_DIR = "./voices"
CACHED_VOICE_TEXT = """Jittery Jack's jam jars jiggled jauntily, jolting Jack's jumbled jelly-filled jars joyously.
Cindy's circular cymbals clanged cheerfully, clashing crazily near Carla's crashing crockery.
You think you can just waltz in here and cause chaos? Well, I've got news for you."""


wav_pattern = os.path.join(CUSTOM_VOICE_DIR, '*.wav')
wav_files = glob.glob(wav_pattern)
if len(wav_files) == 0:
    exit(0)
processed_path = os.path.join(CUSTOM_VOICE_DIR, 'processed')

from einops import rearrange
from voxcpm import VoxCPM
from datetime import datetime
import shutil
import numpy as np


model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

for wav_file in wav_files:
    file_name = os.path.basename(wav_file)
    voice_name = os.path.splitext(file_name)[0]
    print(f"Processing voice: {voice_name}")
    txt_path = os.path.join(CUSTOM_VOICE_DIR, f"{voice_name}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as file:
            prompt_text = file.read()
    else:
        print(f"{voice_name}.txt not found. Using CACHED_VOICE_TEXT")
        prompt_text = CACHED_VOICE_TEXT

    try:
        cache_data = model.tts_model.build_prompt_cache(prompt_text, wav_file)
        audio_feat = rearrange(cache_data['audio_feat'], 't p d -> 1 d p t')
        np.save(os.path.join(VOICE_CACHE_DIR, voice_name), audio_feat)
    except Exception as e:
        print(f"Error converting {voice_name}: {e}")
        continue

    shutil.copy(wav_file, VOICE_CACHE_DIR)
    if os.path.exists(txt_path):
        shutil.copy(txt_path, VOICE_CACHE_DIR)

    # rename old processed wav to avoid collision
    processed_wav = os.path.join(processed_path, file_name)
    if os.path.exists(processed_wav):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(processed_path, f"{voice_name}.{timestamp}.wav")
        os.rename(processed_wav, dest)

    # move wav to processed folder
    shutil.move(wav_file, processed_path)
    print(f"{voice_name} successfully converted")

exit(0)