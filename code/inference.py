import torch
import glob
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import GPT2Config, GPT2Model, TransfoXLConfig, TransfoXLModel
import numpy as np
import pickle
import utils
import os
import argparse
import warnings
warnings.filterwarnings('ignore')




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_input_length', type=int, help='max input length', default=1024)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--temp', type=float, help='temperature', default=1.2)
    parser.add_argument('--topk', type=int, help='topk', default=5)
    parser.add_argument('--embed_dim', type=int, help='embedding dimension', default=512)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension', default=512)
    parser.add_argument('--n_head', type=int, help='number of heads', default=8)
    parser.add_argument('--n_layer', type=int, help='number of layers', default=12)
    parser.add_argument('--dropout', type=float, help='dropout rate', default=0.1)
    parser.add_argument('--device', type=str, help='device to use', default='cuda')
    parser.add_argument('--dict_path', type=str, help='the dictionary path.', default='./dictionary.pkl')
    parser.add_argument('--ckpt_dir', type=str, help='the checkpoint path.', default='')
    parser.add_argument('--output_folder', type=str, help='the output folder.', default='./results')
    args = parser.parse_args()
    return args

opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
## set the input length. must be same with the model config
X_LEN = opt.max_input_length


def extract_events(input_path):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end

    # if you use chord items, you need to add chord_items into "items"
    # e.g. items = tempo_items + note_items + chord_items
    items = tempo_items + note_items

    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.config = TransfoXLConfig(
            vocab_size=len(event2word),
            cutoffs=[],
            d_model=opt.hidden_dim,
            d_embed=opt.embed_dim,
            n_head=opt.n_head,
            n_layer=opt.n_layer,
            dropout=opt.dropout,
            mem_len=512
        )

        # Load the Transformer-XL model
        self.model = TransfoXLModel(self.config)
        
    def forward(self, x, mems=None):
        y = self.model(x, mems)
        return y.last_hidden_state, y.mems

def temperature_sampling(logits, temperature, topk):
    #################################################
    # 1. adjust softmax with the temperature parameter
    # 2. choose top-k highest probs
    # 3. normalize the topk highest probs
    # 4. random choose one from the top-k highest probs as result by the probs after normalize
    #################################################
    logit_after_temp = logits / temperature
    probs = np.exp(logit_after_temp) / np.sum(np.exp(logit_after_temp))
    topk_idx = np.argsort(probs)[-topk:]
    topk_probs = probs[topk_idx]
    topk_probs = topk_probs / topk_probs.sum()
    word = np.random.choice(topk_idx, p = topk_probs, size=1)[0]
    return word
    
def test(opt, n_target_bar = 32, output_folder_dir = '', midi_path = '', prompt = False):
    # check path folder
    try:
        os.makedirs(output_folder_dir, exist_ok=True)
        print(f"dir '{output_folder_dir}' is created")
    except:
        pass

    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    with torch.no_grad():
        # load model
        checkpoint = torch.load(opt.ckpt_dir)
        model = Model(opt).to(opt.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        batch_size = opt.batch_size

        if prompt:  
            # If prompt, load prompt file, extract events, create tokens. (similar to dataset preparation)
            events = extract_events(prompt)
            words = [[event2word['{}_{}'.format(e.name, e.value)] for e in events]]
            words[0].append(event2word['Bar_None'])

        else:  
            # Or, random select prompt to start
            words = []
            for _ in range(batch_size):
                ws = [event2word['Bar_None']]
                if 'chord' in opt.dict_path:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in event2word.items() if 'Chord' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)

        # generate
        mems = None
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 7
        print('Start generating')
        while current_generated_bar < n_target_bar:
            print("\r", current_generated_bar, end="")
            # input
            if initial_flag:
                temp_x = np.zeros((batch_size, original_length))
                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x_new = np.zeros((batch_size, 1))
                for b in range(batch_size):
                    temp_x_new[b][0] = words[b][-1]
                temp_x = np.array([np.append(temp_x[0], temp_x_new[0])])
            
            temp_x = torch.Tensor(temp_x).long()
            
            output_logits, mems = model(temp_x.to(opt.device), mems)
            
            # sampling
            _logit = output_logits[0, -1].to('cpu').detach().numpy()
            word = temperature_sampling(
                logits=_logit, 
                temperature=opt.temp,
                topk=opt.topk)
            
            # print(word)

            words[0].append(word)

            if word == event2word['Bar_None']:
                current_generated_bar += 1

        if prompt:
            utils.write_midi(
                words=words[0][original_length:],
                word2event=word2event,
                output_path=os.path.join(opt.output_folder, midi_path),
                prompt_path=prompt)
        else:
            utils.write_midi(
                words=words[0],
                word2event=word2event,
                output_path=os.path.join(opt.output_folder, midi_path),
                prompt_path=None)
    

def main():
    for prompt in glob.glob('../prompt_song/*.mid'):
        midi_path=f'{prompt.split("/")[-1].split(".")[0]}_generated.mid'
        test(
            opt,
            n_target_bar=32, 
            output_folder_dir=opt.output_folder,
            midi_path=midi_path,
            prompt=prompt)

if __name__ == '__main__':
    main()

