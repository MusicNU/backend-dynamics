from feedback import rms_note_by_note, get_dynamics, get_tempos, load_audio, dynamic_to_rms
import librosa, music21
import numpy as np
from librosa.sequence import dtw
from music21 import converter, dynamics, tempo

def dyanmics_tempos_score_samplerate(filename: str) -> list[music21.stream.Score, list[tuple[float, str]], list[tuple[float, str]], int]:
    sample_rate = 48100
    score: music21.stream.Score = converter.parse(filename)
    dynamics_list: list[tuple[float, str]] = get_dynamics(score)
    tempos_list: list[tuple[float, str]] = get_tempos(score)
    return score, dynamics_list, tempos_list, sample_rate



def test_testing_that_testing_works() -> None:
    score, dynamics_list, tempos_list, sample_rate = dyanmics_tempos_score_samplerate(".\mxl_test_files\\test7.mxl")

    expected_rms: list[float] = []
    time_points: list[float] = []
    expected_rms, time_points = rms_note_by_note(score, dynamics_list, tempos_list, sample_rate)

    assert len(time_points) == len(expected_rms)
    for rms, beat in zip(expected_rms, time_points):
        if beat < 5.0:
            assert rms == dynamic_to_rms.get("mp")
        if beat < 6.0:
            assert rms == rest_db
        
    