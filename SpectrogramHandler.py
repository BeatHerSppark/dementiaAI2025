import os
import librosa
from matplotlib import pyplot as plt
import numpy as np


class SpectrogramHandler:
    def generate_spectrograms(self, audio_folder_path="./DementiaBank/audio", output_folder="./DementiaBank/spectrograms"):
        os.makedirs(f"{output_folder}/Control", exist_ok=True)
        os.makedirs(f"{output_folder}/Dementia", exist_ok=True)
        
        for folder_type in ['Control', 'Dementia']:
            input_folder = f"{audio_folder_path}/{folder_type}/cookie/cleaned"
            output_subfolder = f"{output_folder}/{folder_type}"
            
            if not os.path.exists(input_folder):
                print(f"Warning: {input_folder} not found!")
                continue
            
            mp3_files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]
            print(f"\nProcessing {len(mp3_files)} {folder_type} audio files...")
            
            for i, audio_file in enumerate(mp3_files):
                file_id = audio_file.replace('.mp3', '')
                input_path = f"{input_folder}/{audio_file}"
                output_path = f"{output_subfolder}/{file_id}.png"
                
                if os.path.exists(output_path):
                    print(f"{i+1}/{len(mp3_files)}: {file_id}.png already exists")
                    continue
                
                try:
                    print(f"{i+1}/{len(mp3_files)}: Converting {audio_file}...")
                    
                    # load audio
                    y, sr = librosa.load(input_path, sr=22050)
                    y, _ = librosa.effects.trim(y, top_db=60)
                    
                    # create mel spectrogram
                    n_fft      = 2048
                    hop_length = 256
                    mel_spec = librosa.feature.melspectrogram(
                        y=y, sr=sr, n_mels=128, fmax=8000, n_fft=n_fft, hop_length=hop_length
                    )
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    plt.figure(figsize=(10, 4))
                    plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
                    plt.axis('off')
                    plt.savefig(output_path, format='png', 
                              bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.close()
                    
                    print(f"Saved: {file_id}.png")
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        print(f"\nAll spectrograms saved in: {output_folder}")
        self.show_summary(output_folder)
    
    def show_summary(self, spectrograms_folder):
        control_count = len([f for f in os.listdir(f"{spectrograms_folder}/Control") if f.endswith('.png')])
        dementia_count = len([f for f in os.listdir(f"{spectrograms_folder}/Dementia") if f.endswith('.png')])
        
        print(f"\nSummary:")
        print(f"   Control spectrograms: {control_count}")
        print(f"   Dementia spectrograms: {dementia_count}")
        print(f"   Total: {control_count + dementia_count}")


if __name__ == '__main__':
    spectrogram_handler = SpectrogramHandler()
    spectrogram_handler.generate_spectrograms()
