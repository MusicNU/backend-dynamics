from feedback import rms_note_by_note, get_dynamics, get_tempos, dynamic_to_rms, analyze_performance
import music21, librosa
import numpy as np
from music21 import converter


def test_large_loudness_deviation():
    rms = np.array([0.2, 0.1, 0.9, 0.3, 1.0])  # Some values significantly off
    expected_rms = [librosa.amplitude_to_db([x], ref=np.max)[0] for x in [0.2, 0.4, 0.6, 0.8, 1.0]]
    time_points = [0, 1, 2, 3, 4]

    feedback = analyze_performance(rms, expected_rms, time_points)
    assert len(feedback) > 0, "Expected feedback for large mismatches"
    print("test_large_loudness_deviation:", feedback)

def test_all_mismatches():
    rms = np.array([1.0, 0.9, 0.8, 0.7, 0.6])  # Reversed pattern
    expected_rms = [librosa.amplitude_to_db([x], ref=np.max)[0] for x in [0.2, 0.4, 0.6, 0.8, 1.0]]
    time_points = [0, 1, 2, 3, 4]

    feedback = analyze_performance(rms, expected_rms, time_points)
    print("test_all_mismatches", feedback)
    assert len(feedback) == len(expected_rms), "Expected full feedback for completely wrong loudness"

def test_no_mismatch():
    rms = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    expected_rms = [librosa.amplitude_to_db([x], ref=np.max)[0] for x in [0.2, 0.4, 0.6, 0.8, 1.0]]
    time_points = [0, 1, 2, 3, 4]

    feedback = analyze_performance(rms, expected_rms, time_points)
    assert len(feedback) == 0, "Expected no feedback for a perfect match"
    print("test_no_mismatch: Passed")

def test_slight_mismatch():
    rms = np.array([0.21, 0.39, 0.61, 0.79, 1.01])  # Slight deviation
    expected_rms = [librosa.amplitude_to_db([x], ref=np.max)[0] for x in [0.2, 0.4, 0.6, 0.8, 1.0]]
    time_points = [0, 1, 2, 3, 4]

    feedback = analyze_performance(rms, expected_rms, time_points)
    assert len(feedback) == 0, "Expected no feedback for slight mismatches within threshold"
    print("test_slight_mismatch: Passed")

if __name__ == "__main__":
    test_large_loudness_deviation()
    test_all_mismatches()
    test_no_mismatch()
    test_slight_mismatch()