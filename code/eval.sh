for e in 150; do
    for k in 5 10; do
        for temp in 0.3; do
            python eval_metrics.py \
                --dict_path ./dictionary.pkl \
                --output_file_path ./results_gpt2/results_${e}_${k}_${temp} 
        done
    done
done
