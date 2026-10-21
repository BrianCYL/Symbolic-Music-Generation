from glob import glob
import pandas as pd

def main():
    file = f'./dataset_pop1k7.csv'
    data = pd.read_csv(file)
    data = data.drop(columns=['piece_name'])
    print(f'Dataset')
    print(data.mean())

    for e in ['050', '100', '150']:
        for k in [5, 10, 30]:
            for temp in [0.3, 0.7, 0.9]:
                file = f'./metrics_gpt2/results_{e}_{k}_{temp}_pop1k7.csv'
                data = pd.read_csv(file)
                data = data.drop(columns=['piece_name'])
                print(f'Epoch: {e}, top-k: {k}, Temp: {temp}')
                print(data.mean())
    

if __name__ == '__main__':
    main()

