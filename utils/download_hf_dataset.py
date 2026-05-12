"""
Download a TTS dataset from Hugging Face and save audio files + metadata

Example Usage
-------------
python3 download_hf_dataset.py
"""

import os
import csv
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf

HF_TOKEN = os.getenv("HF_TOKEN")

def download_tts_dataset(dataset_name, dest_folder, split="train", max_samples=None):
    """
    Download a TTS dataset from Hugging Face and save audio files + metadata
    """
    
    # Create destination folders
    audio_folder = os.path.join(dest_folder, "wavs")
    os.makedirs(audio_folder, exist_ok=True)
    
    # Dataset fields
    audio_field = "audio"
    text_field = "transcription"
    speaker_field = "speaker"
    gender_field = "gender"
    
    print(f"Loading dataset: {dataset_name} - {split} split...")
    
    # Load the dataset with the specified subset
    dataset = load_dataset(dataset_name, split=split, token=HF_TOKEN)
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Total samples to download: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    
    # Prepare metadata CSV
    csv_path = os.path.join(dest_folder, "metadata.csv")
    
    # Check which optional fields are available
    available_fields = dataset.features.keys()
    has_speaker = speaker_field in available_fields
    has_gender = gender_field in available_fields
    
    # Prepare CSV header
    csv_headers = ["filename", "transcription", "speaker", "gender"]
    
    # Process and save dataset
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers, delimiter='|')
        writer.writeheader()
        
        for idx, sample in enumerate(tqdm(dataset, desc="Downloading audio files")):
            # Generate filename
            filename = f"audio_{idx:05d}.wav"
            audio_path = os.path.join(audio_folder, filename)
            
            # Extract audio data
            audio_data = sample[audio_field]
            audio_array = audio_data["array"]
            sampling_rate = audio_data["sampling_rate"]
            
            # Save audio file
            sf.write(str(audio_path), audio_array, sampling_rate)
            
            # Prepare metadata row
            row = {
                "filename": filename,
                "transcription": str(sample[text_field]),
                "speaker": str(sample[speaker_field]) if has_speaker else "",
                "gender": str(sample[gender_field]) if has_gender else "",
            }
            
            writer.writerow(row)
    
    print(f"\n✓ Download complete!")
    print(f"✓ Audio files saved to: {audio_folder}")
    print(f"✓ Metadata saved to: {csv_path}")
    print(f"✓ Total files: {len(dataset)}")


if __name__ == "__main__":
    download_tts_dataset(
        dataset_name="aimerou/kallaama-bench-filtered",
        dest_folder="/root/kallaama-bench/dev",
        split="train"
    )