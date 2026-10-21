python ./inference.py \
    --max_input_length 512 \
    --batch_size 1 \
    --temp 0.3 \
    --topk 5 \
    --dict_path ../dictionary.pkl \
    --ckpt_dir ../checkpoints/transformXL_checkpoint.pkl \
    --output_folder ${1}/config1

python ./inference.py \
    --max_input_length 512 \
    --batch_size 1 \
    --temp 0.7 \
    --topk 10 \
    --dict_path ../dictionary.pkl \
    --ckpt_dir ../checkpoints/transformXL_checkpoint.pkl \
    --output_folder ${1}/config2

python ./inference.py \
    --max_input_length 512 \
    --batch_size 1 \
    --temp 0.9 \
    --topk 30 \
    --dict_path ../dictionary.pkl \
    --ckpt_dir ../checkpoints/transformXL_checkpoint.pkl \
    --output_folder ${1}/config3

