from feedback import get_dynamics, get_tempos, load_audio, dynamic_to_rms
import librosa, music21
import numpy as np
from librosa.sequence import dtw
from music21 import converter, dynamics, tempo

def dyanmics_tempos_score_samplerate(filename: str) -> list[music21.stream.Score, list[tuple[float, str]], list[tuple[float, str]], int]:
    sample_rate = 48100
    sample_rate = 100
    score: music21.stream.Score = converter.parse(filename)
    dynamics_list: list[tuple[float, str]] = get_dynamics(score)
    tempos_list: list[tuple[float, str]] = get_tempos(score)
    return score, dynamics_list, tempos_list, sample_rate

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

def test_rms_note_by_note() -> None:
    def verifier(correct_beat_and_dynamic: list[tuple[float, str]], expected_rms: list[float], time_points: list[float]) -> None:
        correct_ptr: int = 0
        for rms, beat in zip(expected_rms, time_points):
            if beat >= correct_beat_and_dynamic[correct_ptr][0]:
                correct_ptr += 1
            assert rms == dynamic_to_rms.get(correct_beat_and_dynamic[correct_ptr][1])

    score, dynamics_list, tempos_list, sample_rate = dyanmics_tempos_score_samplerate(".\mxl_test_files\\test7.mxl")
    expected_rms, time_points = rms_note_by_note(score, dynamics_list, tempos_list, sample_rate)
    correct_beat_and_dynamic: list[tuple[float, str]] = [
        (5.0, "mp"), (6.0, "rest"), (7.0, "mp"), (8.0, "rest"), (9.0, "mp"), (10.0, "rest"), (11.0, "mp"), (12.0, "rest"), (13.0, "ff"), (15.0, "rest"), (17.0, "ff"), (19.0, "rest"), (21.0, "ff"), (24.0, "rest")
    ]
    assert len(time_points) == len(expected_rms)
    verifier(correct_beat_and_dynamic, expected_rms, time_points)  


    score, dynamics_list, tempos_list, sample_rate = dyanmics_tempos_score_samplerate(".\\mxl_test_files\\test8.mxl")
    expected_rms, time_points = rms_note_by_note(score, dynamics_list, tempos_list, sample_rate)
    correct_beat_and_dynamic: list[tuple[float, str]] = [
        (4.0, "mf"), (8.0, "rest"), (12.0, "pp"), (16.0, "rest")
    ]
    assert len(time_points) == len(expected_rms)
    verifier(correct_beat_and_dynamic, expected_rms, time_points)  


test_rms_note_by_note()
    