import librosa
import numpy as np

def calculate_dB_levels(audio_file_path, frame_length=2048, hop_length=512): #change lengths as needed
    # Load the audio file at its native sampling rate
    audio, sr = librosa.load(audio_file_path, sr=None)

    # Compute RMS energy for each frame
    rms_energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)

    rms_dB = librosa.amplitude_to_db(rms_energy, ref=np.max)

    # Flatten the array to make it easier to read and process
    rms_dB_flat = rms_dB.flatten()
    
    return rms_dB_flat

if __name__ == "__main__":
    audio_file = '/Users/richardwu/Desktop/MusicNU Dynamics/Ode to Joy - Beethoven (Easy variation) [Piano Sheet Music].wav' #filepath
    dB_levels = calculate_dB_levels(audio_file)

    # print db
    for dB in dB_levels:
        print(f"dB: {dB:.2f} dB")