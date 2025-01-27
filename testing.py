import librosa, music21
import numpy as np
from librosa.sequence import dtw
from feedback_functions import get_dynamics, get_tempos
from music21 import *
from typing import List, Tuple

def rms_note_by_note(score: music21.stream.Score, dynamics_list: List[Tuple[float, str]], tempos_list: List[Tuple[float, str]], sample_rate: int) -> tuple[list, list]:
    def extend_expected_rms(start: float, end: float, decibel: int, tempo: int) -> None:
        duration: float = end - start
        samples: int = int((duration / tempo) * sample_rate)
        expected_rms.extend([decibel] * samples)
        time_points.extend(np.linspace(start, end, samples))

    expected_rms: list[float] = []
    time_points: list[float] = []
    note_ptr, dyn_ptr, tempo_ptr = 0, 0, 0
    cur_beat = 0.0
    dynamic_to_rms = {
        "pp": -40, "p": -30, "mp": -25,
        "mf": -20, "f": -10, "ff": 0
    }

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
        cur_dyn = dynamic_to_rms.get(dynamics_list[dyn_ptr][1]) if note.isNote else -80

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

def main():
    score: music21.stream.Score = converter.parse("test6.mxl")
    dynamics_list: List[Tuple[float, str]] = get_dynamics(score)
    tempos_list: List[Tuple[float, str]] = get_tempos(score)
    expected_rms: list[float]
    time_points: list[float]
    expected_rms, time_points = rms_note_by_note(score, dynamics_list, tempos_list, 100)

    for i in range(len(expected_rms)):
        print(f"At time {time_points[i]:.2f}, should play with {expected_rms[i]:.2f} dB.")

if __name__ == '__main__':
    main()