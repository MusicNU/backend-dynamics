import librosa, music21
import numpy as np
from librosa.sequence import dtw
from music21 import converter, dynamics, tempo

dynamic_to_rms: dict[str, int] = {
    "pp": -40, "p": -30, "mp": -25,
    "mf": -20, "f": -10, "ff": 0, 
    "rest": -80,        # setting rest db to -80
    "default": -20      # if no dynamic is given, default sets to -20db
}

#if no tempo is provided in the score, default to 120bpm
default_tempo: int = 120

def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio file and calculate RMS."""
    y, sample_rate = librosa.load(audio_path)
    rms = librosa.feature.rms(y=y)[0]
    return rms, sample_rate

def get_dynamics(score: music21.stream.Score) -> list[tuple[float, str]]:
    """Extract dynamics markings from score."""
    sheet_dynamics = []
    for element in score.flatten():
        if isinstance(element, dynamics.Dynamic):
            sheet_dynamics.append((element.offset, element.value))

    # if the dynamic is not provided for the score, or the dynamic is not provided until mid-way through the music
    if not sheet_dynamics or sheet_dynamics[0][0] > 0.0:
        sheet_dynamics.insert(0, (0.0, "default"))
    return sheet_dynamics

def get_tempos(score: music21.stream.Score) -> list[tuple[float, int]]:
    """Get tempo in beats per second."""
    tempos = []
    for metronome_mark in score.flatten().getElementsByClass(tempo.MetronomeMark):
        tempos.append((metronome_mark.offset, metronome_mark.number))
        
    # if the tempo is not provided for the score, or the tempo is not provided until mid-way through the music
    if not tempos or tempos[0][0] > 0.0:
        tempos.insert(0, (0.0, default_tempo))
    return tempos

def rms_note_by_note(score: music21.stream.Score, dynamics_list: list[tuple[float, str]], tempos_list: list[tuple[float, str]], sample_rate: int) -> tuple[list, list]:

    def extend_expected_rms(start: float, end: float, decibel: int, tempo: int) -> None:
        duration: float = end - start
        samples: int = int((duration / tempo) * sample_rate)
        expected_rms.extend([decibel] * samples)
        time_points.extend(np.linspace(start, end, samples, endpoint=False))

    expected_rms: list[float] = []
    time_points: list[float] = []
    note_ptr, dyn_ptr, tempo_ptr = 0, 0, 0
    cur_beat = 0.0

    # only consider the first part in the score, for now at least
    notes_and_rests = list(score.parts[0].recurse().notesAndRests)

    while note_ptr < len(notes_and_rests):
        note = notes_and_rests[note_ptr]

        note_length: float = note.duration.quarterLength
        next_note_change: float = cur_beat + note_length
        next_dynamic_change: float = dynamics_list[dyn_ptr + 1][0] if dyn_ptr < len(dynamics_list) - 1 else float('inf')
        next_tempo_change: float = tempos_list[tempo_ptr + 1][0] if tempo_ptr < len(tempos_list) - 1 else float('inf')

        cur_end: float = next_note_change
        cur_tempo: int = tempos_list[tempo_ptr][1]
        cur_dyn = dynamic_to_rms.get(dynamics_list[dyn_ptr][1]) if note.isNote else dynamic_to_rms.get("rest")

        if(next_dynamic_change < next_note_change):
            cur_end = next_dynamic_change
            dyn_ptr += 1
        else:
            if next_tempo_change == next_note_change:
                tempo_ptr += 1
            note_ptr += 1

        extend_expected_rms(start=cur_beat, end=cur_end, decibel=cur_dyn, tempo=cur_tempo)
        cur_beat = min(next_note_change, next_dynamic_change, next_tempo_change)
        
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

def main():
    audio_path = "user_recording.wav"
    sheet_music_path = "sheet_music.xml"

    rms, sample_rate = load_audio(audio_path)
    
    score = converter.parse(sheet_music_path)
    dynamics_list = get_dynamics(score)
    tempo_list = get_tempos(score)
    
    expected_rms, time_points = rms_note_by_note(score, dynamics_list, tempo_list, sample_rate)
    
    feedback = analyze_performance(rms, expected_rms, time_points)
    
    # Print results
    print("\n".join(feedback))

if __name__ == "__main__":
    main()