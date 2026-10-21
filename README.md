# Symbolic Music Generation

This repository implements symbolic music generation with a Transformer-XL
model. MIDI files are converted into event tokens, and the model generates
continuations from prompt songs. Generated MIDI files can also be rendered as
WAV audio with a SoundFont.

## Setup

Install the dependencies listed by running:

```bash
pip install -r requirements.txt
```

Download the [dataset](https://drive.google.com/drive/folders/1E5Xrk5TN2sPXYz1idZYxVIEterZpqwyY?usp=sharing), [dictionary](https://drive.google.com/file/d/1op--djXmPV86QJ7Ayx044yZoHB5-JG0E/view?usp=sharing), [prompt song](https://drive.google.com/drive/folders/1JveK9tin7IYxhSyQj8EzCtJ7TFk-7c_0?usp=sharing) and the
`GeneralUser-GS.sf2` SoundFont from 
[here](https://drive.google.com/file/d/1IziuDULljB_8xAJTcyhk_ylAAmB2o11Q/view?usp=sharing).

Run commands from the `Symbolic-Music-Generation/` directory unless otherwise specified. The
expected directory layout is:

```text
Symbolic-Music-Generation/
├── code/
│   ├── ...
│   ├── inference.py
│   ├── midi2wav.py
│   ├── train.py
│   └── utils.py
├── dataset/
│   └── *.mid
├── prompt_song/
│   ├── song_1.mid
│   ├── song_2.mid
│   └── song_3.mid
├── checkpoints/
│   └── transformXL_checkpoint.pkl
├── dictionary.pkl
└── GeneralUser-GS.sf2
```

Please note that:

- Prompt MIDI files must be stored in `./prompt_song/`.
- The inference script searches for files matching `../prompt_song/*.mid` when
  run from `./code/`.
- Training MIDI files are read from `./dataset/*.mid`.
- `dictionary.pkl` must match the vocabulary used to train the checkpoint.
- `GeneralUser-GS.sf2` is required only for MIDI-to-WAV conversion.
- If you only need to run inference, one can download the checkpoints from [here](https://drive.google.com/drive/folders/1E5Xrk5TN2sPXYz1idZYxVIEterZpqwyY?usp=sharing). 

## Model 1: Transformer-XL

### Training

Train the Transformer-XL model on the MIDI files in `./dataset/` with the
default settings (100 epochs, batch size 4, learning rate `0.001`, 12 layers,
and 5 grouped segments):

```bash
python code/train.py
```

The training script saves checkpoints every 10 epochs and writes the training
loss to `./checkpoints/training_loss.npy`.

To use the settings from the provided training script:

```bash
python code/train.py \
  --n_epochs 150 \
  --max_input_length 512 \
  --lr 1e-3 \
  --batch_size 16 \
  --ckpt_folder ./REMI_BPE_checkpoints \
  --dict_path ./dictionary.pkl
```

To resume training, pass a previous checkpoint with `--ckpt_dir` and keep the
model configuration consistent with the checkpoint.

### Inference

The inference script generates a 32-bar continuation for every MIDI file in
`./prompt_song/`. Run it from the `code/` directory because the script uses
paths relative to that directory:

```bash
cd code
python inference.py \
  --max_input_length 512 \
  --batch_size 1 \
  --temp 0.3 \
  --topk 5 \
  --dict_path ../dictionary.pkl \
  --ckpt_dir ../checkpoints/transformXL_checkpoint.pkl \
  --output_folder ../results
cd ..
```

The generated files are saved as:

```text
results/
├── song_1_generated.mid
├── song_2_generated.mid
└── song_3_generated.mid
```

The `--temp` option controls sampling randomness, while `--topk` limits
sampling to the most probable tokens. Higher values generally produce more
variation.

The repository also provides three example configurations:

```bash
cd code
python inference.py --max_input_length 512 --batch_size 1 \
  --temp 0.3 --topk 5 \
  --dict_path ../dictionary.pkl \
  --ckpt_dir ../checkpoints/transformXL_checkpoint.pkl \
  --output_folder ../config1

python inference.py --max_input_length 512 --batch_size 1 \
  --temp 0.7 --topk 10 \
  --dict_path ../dictionary.pkl \
  --ckpt_dir ../checkpoints/transformXL_checkpoint.pkl \
  --output_folder ../config2

python inference.py --max_input_length 512 --batch_size 1 \
  --temp 0.9 --topk 30 \
  --dict_path ../dictionary.pkl \
  --ckpt_dir ../checkpoints/transformXL_checkpoint.pkl \
  --output_folder ../config3
cd ..
```

## Model 2: GPT-2 Method

### Training

Train the GPT-2 MIDI language model on the MIDI files in `./dataset/` with
the default settings (100 epochs, batch size 4, learning rate `0.001`, 12
layers, and 5 grouped segments):

```bash
python code/train_gpt.py
```

GPT-2 checkpoints are saved every 10 epochs in `./checkpoints/`, and the
training loss is saved to `./checkpoints/training_loss.npy`. To use a separate
checkpoint directory or change the model settings:

```bash
python code/train_gpt.py \
  --n_epochs 150 \
  --max_input_length 512 \
  --lr 1e-3 \
  --batch_size 16 \
  --ckpt_folder ./GPT2_checkpoints \
  --dict_path ./dictionary.pkl
```

The GPT-2 model configuration must match the configuration used during
inference, including `max_input_length`, `hidden_dim`, `embed_dim`, `n_head`,
`n_layer`, and `dropout`.

### Inference

Generate a 32-bar continuation for every MIDI file in `./prompt_song/` using a
GPT-2 checkpoint. Run the command from the `code/` directory because the
inference script uses paths relative to that directory:

```bash
cd code
python inference_gpt.py \
  --max_input_length 512 \
  --batch_size 1 \
  --temp 0.7 \
  --topk 10 \
  --dict_path ../dictionary.pkl \
  --ckpt_dir ../GPT2_checkpoints/epoch_100.pkl \
  --output_folder ../results_gpt2
cd ..
```

The generated MIDI files are saved in the specified output directory:

```text
results_gpt2/
├── song_1_generated.mid
├── song_2_generated.mid
└── song_3_generated.mid
```

The `--temp` option controls sampling randomness, while `--topk` restricts
sampling to the top-k most probable events. For example, the provided
sampling configurations use `(temperature, top-k)` values of `(0.3, 5)`,
`(0.7, 10)`, and `(0.9, 30)`.

## Convert MIDI files to WAV

After generating MIDI files, render all MIDI files in an output directory with
FluidSynth:

```bash
python code/midi2wav.py \
  --sf_dir ./GeneralUser-GS.sf2 \
  --midi_folder_dir ./results
```

Each generated `.mid` file receives a corresponding `.wav` file in the same
directory.
