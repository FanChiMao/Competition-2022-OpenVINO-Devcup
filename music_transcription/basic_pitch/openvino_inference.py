from openvino.inference_engine import IECore
from music_transcription.basic_pitch.utils.note_creation import *
from music_transcription.basic_pitch.utils.audio_to_frame import *
from music_transcription.basic_pitch.utils.write_data import *
from music_transcription.basic_pitch.utils.color_text import *
import argparse

from music_transcription.basic_pitch.constants import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # ignore the message

def music_transcription(input_audio, save_dictionary):
    model_onnx = "./music_transcription/models/basic_pitch_43844_model.onnx"

    print(toYellow("--------------- Music transcription ---------------"))

    print(toYellow("==> Start transcription the separated audio: "))
    print(toYellow("    " + input_audio))
    ie = IECore()

    print(toYellow('  Loading onnx files from:\n    {}'.format(model_onnx)))
    network = ie.read_network(model=model_onnx)

    print(toYellow('  Preprcoess audio to 2D tensor frame'))
    audio_windowed, _, audio_original_length = get_audio_input(input_audio)

    print(toYellow('  Set input audio frame tensor to: ({}, {}, {})'.format(audio_windowed.shape[0], AUDIO_N_SAMPLES, 1)))
    network.reshape({'input': (audio_windowed.shape[0], AUDIO_N_SAMPLES, 1)})

    print(toYellow('  Loading models to the plugin'))
    exec_net = ie.load_network(network=network, device_name="GPU")

    # print(toYellow('==> Starting inference!'))
    output = exec_net.infer(inputs={'input': audio_windowed})

    unwrapped_output = {k: unwrap_output(output[k], audio_original_length, n_overlapping_frames) for k in output}
    min_note_len = int(np.round(58 / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))  # minimum_note_length: 58

    print(toYellow('  Generate midi data'))
    midi_data, note_event = model_output_to_notes(
        output=unwrapped_output,
        onset_thresh=0.5,
        frame_thresh=0.3,
        min_note_len=min_note_len,  # convert to frames
        min_freq=None,
        max_freq=None,
        multiple_pitch_bends=False,
        melodia_trick=True,
        )

    # print(toYellow('==> Save midi data to {}'.format(save_dictionary)))
    write_midi_file(input_audio, os.path.join(save_dictionary, "transcription_results"), midi_data)

    print(toYellow("--------------- Music transcription ---------------"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run basic pitch inference via OpenVINO')
    parser.add_argument('--input_audio', default='./sample_audio/vocals.wav', type=str, help='input audio file path')
    parser.add_argument('--model_onnx', default='./basic_pitch_43844_model.onnx', type=str, help='onnx models path')
    parser.add_argument('--save_dictionary', default='./result', type=str, help='saving result path')

    args = parser.parse_args()

    music_transcription(input_audio=args.input_audio, save_dictionary=args.save_dictionary)
