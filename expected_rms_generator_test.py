from feedback import rms_note_by_note, get_dynamics, get_tempos, dynamic_to_rms
import music21
from music21 import converter

def dyanmics_tempos_score_samplerate(filename: str) -> list[music21.stream.Score, list[tuple[float, str]], list[tuple[float, str]], int]:
    sample_rate = 48100
    score: music21.stream.Score = converter.parse(filename)
    dynamics_list: list[tuple[float, str]] = get_dynamics(score)
    tempos_list: list[tuple[float, str]] = get_tempos(score)
    return score, dynamics_list, tempos_list, sample_rate

def test_rms_note_by_note() -> None:
    def verifier(correct_beat_and_dynamic: list[tuple[float, str]], expected_rms: list[float], time_points: list[float]) -> None:
        correct_ptr: int = 0
        for rms, beat in zip(expected_rms, time_points):
            if beat >= correct_beat_and_dynamic[correct_ptr][0]:
                correct_ptr += 1
            if (rms != dynamic_to_rms.get(correct_beat_and_dynamic[correct_ptr][1])):
                print(f"error, beat is {beat}, rms is {rms}, but expected is {dynamic_to_rms.get(correct_beat_and_dynamic[correct_ptr][1])}")
            assert rms == dynamic_to_rms.get(correct_beat_and_dynamic[correct_ptr][1])

    score, dynamics_list, tempos_list, sample_rate = dyanmics_tempos_score_samplerate(".\\mxl_test_files\\test7.mxl")
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

    score, dynamics_list, tempos_list, sample_rate = dyanmics_tempos_score_samplerate(".\\mxl_test_files\\test9.mxl")
    expected_rms, time_points = rms_note_by_note(score, dynamics_list, tempos_list, sample_rate)
    correct_beat_and_dynamic: list[tuple[float, str]] = [
        (1.5, "p"), (3.0, "rest"), (3.125, "p"), (3.25, "rest"), (3.375, "mf"), (3.5, "rest"), (8.0, "mf"), (12.0, "f"), (24.0, "rest")
    ]
    assert len(time_points) == len(expected_rms)
    verifier(correct_beat_and_dynamic, expected_rms, time_points)  


test_rms_note_by_note()