FFT_HOP = 256
N_FFT = 8 * 256

NOTES_BINS_PER_SEMITONE = 1
CONTOURS_BINS_PER_SEMITONE = 3

ANNOTATIONS_BASE_FREQUENCY = 27.5  # lowest key on a piano
ANNOTATIONS_N_SEMITONES = 88  # number of piano keys
AUDIO_SAMPLE_RATE = 22050 # 22050 44100
AUDIO_N_CHANNELS = 1
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE

AUDIO_WINDOW_LENGTH = 2  # duration in seconds of training examples - original 1

ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
ANNOTATION_HOP = 1.0 / ANNOTATIONS_FPS

# ANNOT_N_TIME_FRAMES is the number of frames in the time-frequency representations we compute
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH

# AUDIO_N_SAMPLES is the number of samples in the (clipped) audio that we use as input to the models
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP # 87944

n_overlapping_frames = 30
overlap_len = n_overlapping_frames * FFT_HOP # 7680
hop_size = AUDIO_N_SAMPLES - overlap_len # 87944 - 7680 = 80264 // 43844

MAX_FREQ_IDX = 87
N_PITCH_BEND_TICKS = 8192
MIDI_OFFSET = 21
SONIFY_FS = 3000
