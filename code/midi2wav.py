from midi2audio import FluidSynth
from glob import glob
import os
import argparse

def midi2audio(args):
# Specify the path to your SoundFont file (downloaded separately)
    soundfont_path = args.sf_dir
    midi_list = glob(os.path.join(args.midi_folder_dir, '*.mid'))

    # Initialize the synthesizer with the SoundFont
    fs = FluidSynth(soundfont_path)

    # Convert a MIDI file to WAV
    for midi_file in midi_list:
        wav_file = midi_file.replace('.mid', '.wav')
        fs.midi_to_audio(midi_file, wav_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sf_dir', type=str, default='./GeneralUser-GS.sf2')
    parser.add_argument('--midi_folder_dir', type=str, default='./results_gpt2/results_100_5_0.7')
    args = parser.parse_args()
    midi2audio(args)

if __name__ == '__main__':
    main()