import librosa
import numpy as np
from librosa.sequence import dtw
from music21 import converter, dynamics, tempo


# Load audio and calculate RMS
y, sr = librosa.load("user_recording.wav")
rms = librosa.feature.rms(y=y)[0]


# Extract dynamics and tempo from sheet music
score = converter.parse("sheet_music.xml")
sheet_dynamics = []
for element in score.flat:
   if isinstance(element, dynamics.Dynamic):
       sheet_dynamics.append((element.offset, element.value))


tempo_marking = score.flat.getElementsByClass(tempo.MetronomeMark)[0]
tempo_bpm = tempo_marking.number
beats_per_second = tempo_bpm / 60


# Generate expected RMS curve from sheet music dynamics
expected_rms = []
time_points = []


for i, (start_offset, dynamic) in enumerate(sheet_dynamics[:-1]):
   next_offset = sheet_dynamics[i + 1][0]
   duration = next_offset - start_offset
   dynamic_rms = {
       "pp": -40, "p": -30, "mp": -25,
       "mf": -20, "f": -10, "ff": 0
   }.get(dynamic, -30)
   samples = int((duration / beats_per_second) * sr)
   expected_rms.extend([dynamic_rms] * samples)
   time_points.extend(np.linspace(start_offset, next_offset, samples))


# Perform DTW alignment
path, cost = dtw(rms, np.array(expected_rms), subseq=True)


# Check alignment and feedback
feedback = []
for i, j in path:
   actual_db = librosa.amplitude_to_db([rms[i]])[0]
   expected_db = expected_rms[j]
   if abs(actual_db - expected_db) > 5:  # tolerance
       feedback.append(f"Mismatch at time {time_points[j]}s: "
                       f"Expected {expected_db} dB, got {actual_db} dB.")


print("\n".join(feedback))
