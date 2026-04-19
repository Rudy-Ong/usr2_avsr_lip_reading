import pandas as pd
import os
import subprocess

df = pd.read_csv("csv/your_filepath_here.csv") 

AUDIO_DIR = "data/audio" 
os.makedirs(AUDIO_DIR, exist_ok=True)

for idx, row in df.iterrows():
    video_path = row["video_path"] 
    if os.path.exists(video_path):
        parts = video_path.split("/")

        # Get the second to last item for the directory
        parent_dir = parts[-2]

        # Get the last item and remove the extension for the filename
        filename = parts[-1].split(".")[0]
        
        output_path = f"{parent_dir}_{filename}.wav"
        audio_path = os.path.join(AUDIO_DIR, output_path)
        subprocess.run([
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0", "-map", "a", "-y", audio_path,
        ], check=True)
    else:
        print(f"Video file not found: {video_path}")