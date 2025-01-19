import librosa
import music21
import numpy as np
from librosa.sequence import dtw
from music21 import converter, dynamics, tempo

def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio file and calculate RMS."""
    y, sr = librosa.load(audio_path)
    rms = librosa.feature.rms(y=y)[0]
    return rms, sr

def get_dynamics(score: music21.stream.Score) -> list[tuple[float, str]]:
    """Extract dynamics markings from score."""
    sheet_dynamics = []
    for element in score.flat:
        if isinstance(element, dynamics.Dynamic):
            sheet_dynamics.append((element.offset, element.value))
    return sheet_dynamics

def get_tempo(score: music21.stream.Score) -> float:
    """Get tempo in beats per second."""
    tempo_marking = score.flat.getElementsByClass(tempo.MetronomeMark)[0]
    tempo_bpm = tempo_marking.number
    return tempo_bpm / 60

def generate_expected_rms(sheet_dynamics: list, beats_per_second: float, sr: int) -> tuple[list, list]:
    """Generate expected RMS curve from sheet music dynamics."""
    dynamic_to_rms = {
        "pp": -40, "p": -30, "mp": -25,
        "mf": -20, "f": -10, "ff": 0
    }
    
    expected_rms = []
    time_points = []
    
    for i, (start_offset, dynamic) in enumerate(sheet_dynamics[:-1]):
        next_offset = sheet_dynamics[i + 1][0]
        duration = next_offset - start_offset
        dynamic_rms = dynamic_to_rms.get(dynamic, -30)
        
        samples = int((duration / beats_per_second) * sr)
        expected_rms.extend([dynamic_rms] * samples)
        time_points.extend(np.linspace(start_offset, next_offset, samples))
        
    return expected_rms, time_points

def analyze_performance(rms: np.ndarray, expected_rms: list, time_points: list) -> list[str]:
    """Analyze performance and generate feedback."""
    path, _ = dtw(rms, np.array(expected_rms), subseq=True)
    
    feedback = []
    for i, j in path:
        actual_db = librosa.amplitude_to_db([rms[i]])[0]
        expected_db = expected_rms[j]
        if abs(actual_db - expected_db) > 5:  # tolerance
            feedback.append(
                f"Mismatch at time {time_points[j]:.2f}s: "
                f"Expected {expected_db:.1f} dB, got {actual_db:.1f} dB."
            )
    return feedback

def main(audio_path: str = "user_recording.wav", sheet_music_path: str = "sheet_music.xml"):
    # Load and process audio
    rms, sr = load_audio(audio_path)
    
    # Load and process score
    score = converter.parse(sheet_music_path)
    sheet_dynamics = get_dynamics(score)
    beats_per_second = get_tempo(score)
    
    # Generate expected RMS curve
    expected_rms, time_points = generate_expected_rms(sheet_dynamics, beats_per_second, sr)
    
    # Analyze and get feedback
    feedback = analyze_performance(rms, expected_rms, time_points)
    
    # Print results
    print("\n".join(feedback))

if __name__ == "__main__":
    main()