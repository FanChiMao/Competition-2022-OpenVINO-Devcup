import librosa
from tensorflow import signal
import numpy as np

from music_transcription.basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    ANNOTATIONS_FPS,
    AUDIO_N_SAMPLES,
    ANNOT_N_FRAMES,
    FFT_HOP,
    overlap_len,
    hop_size)

def unwrap_output(output, audio_original_length, n_overlapping_frames):

    raw_output = output# .numpy()
    if len(raw_output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        raw_output = raw_output[:, n_olap:-n_olap, :]

    output_shape = raw_output.shape
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped_output = raw_output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]  # trim to original audio length

def model_frames_to_time(n_frames: int) -> np.ndarray:
    original_times = librosa.core.frames_to_time(
        np.arange(n_frames),
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
    )
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + 0.0018  # this is a magic number, but it's needed for this to align properly
    times = original_times - (window_offset * window_numbers)
    return times

def window_audio_file(audio_original, hop_size):
    from tensorflow import expand_dims  # imporing this here so the module loads faster

    audio_windowed = expand_dims(
        signal.frame(audio_original, AUDIO_N_SAMPLES, hop_size, pad_end=True, pad_value=0),
        axis=-1,
    )
    window_times = [
        {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        for t_start in np.arange(audio_windowed.shape[0]) * hop_size / AUDIO_SAMPLE_RATE
    ]
    return audio_windowed, window_times


def get_audio_input(audio_path: str):
    assert overlap_len % 2 == 0, "overlap_length must be even, got {}".format(overlap_len)

    audio_original, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)

    original_length = audio_original.shape[0]
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    audio_windowed, window_times = window_audio_file(audio_original, hop_size)
    return audio_windowed, window_times, original_length