from music_source_separation.umx_openvino.run_umx_openvino import *
from music_transcription.basic_pitch.openvino_inference import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run converter by OpenVINO')
    parser.add_argument('--input_audio', default='./sample_audio/CountingStars.wav', help='input audio file path')
    parser.add_argument('--type', default='vocals', help='separation type', choices=['bass', 'drum', 'other', 'vocals'])
    parser.add_argument('--result_dir', default='./result', help='saving result path')

    args = parser.parse_args()

    selected_path = music_source_separation(args.input_audio, args.result_dir, args.type)
    # selected_path = "result/separation_results/TalkLove/vocals.mp3"
    music_transcription(selected_path, args.result_dir)