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

# class NewsDataset(Dataset):
#     def __init__(self, midi_l = [], prompt = ''):
#         self.midi_l = midi_l
#         self.x_len = X_LEN
#         self.dictionary_path = opt.dict_path
#         self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
#         self.parser = self.prepare_data(self.midi_l)
    
#     def __len__(self):
#         return len(self.parser)  
    
#     def __getitem__(self, index):
#         return self.parser[index]
    
#     def chord_extract(self, midi_path, max_time):
#         ####################################################
#         # add your chord extraction method here if you want
#         ####################################################
#         return
    
    # def extract_events(self, input_path):
    #     note_items, tempo_items = utils.read_items(input_path)
    #     note_items = utils.quantize_items(note_items)
    #     max_time = note_items[-1].end

    #     # if you use chord items, you need to add chord_items into "items"
    #     # e.g. items = tempo_items + note_items + chord_items
    #     items = tempo_items + note_items

    #     groups = utils.group_items(items, max_time)
    #     events = utils.item2event(groups)
    #     return events
        
    # def prepare_data(self, midi_paths):
    #     # extract events
    #     all_events = []
    #     for path in midi_paths:
    #         events = self.extract_events(path)
    #         all_events.append(events)
    #     # event to word
    #     all_words = []
    #     for events in all_events:
    #         words = []
    #         for event in events:
    #             e = '{}_{}'.format(event.name, event.value)
    #             if e in self.event2word:
    #                 words.append(self.event2word[e])
    #             else:
    #                 # OOV
    #                 if event.name == 'Note Velocity':
    #                     # replace with max velocity based on our training data
    #                     words.append(self.event2word['Note Velocity_21'])
    #                 else:
    #                     # something is wrong
    #                     # you should handle it for your own purpose
    #                     print('something is wrong! {}'.format(e))
    #         all_words.append(words)
        
    #     # all_words is a list containing words list of all midi files
    #     # all_words = [[tokens of midi], [tokens of midi], ...]

    #     # you can cut the data into what you want to feed into model
    #     # Warning : this example cannot use in transformer_XL, you must implement group segments by yourself
    #     segments = []
    #     for words in all_words:
    #         pairs = []
    #         for i in range(0, len(words)-self.x_len-1, self.x_len):
    #             x = words[i:i+self.x_len]
    #             y = words[i+1:i+self.x_len+1]
    #             pairs.append([x, y])
    #         # abandon last segments in a midi
    #         pairs = pairs[0:len(pairs)-(len(pairs)%5)]
    #         segments = segments + pairs

    #     segments = np.array(segments)
    #     print(segments.shape)
    #     return segments

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
        self.config = GPT2Config(
            vocab_size=len(event2word),
            n_positions=opt.max_input_length,
            n_embd=opt.embed_dim,
            n_head=opt.n_head,
            n_layer=opt.n_layer,
        )

        # Load the Transformer-XL model
        self.model = GPT2Model(self.config)
        
    def forward(self, x):
        y = self.model(x)
        return y.last_hidden_state

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

    # deal wtih NaN values
    flag = 0
    if np.isnan(topk_probs).any():
        flag = 1
        print("NaN values found in probabilities, replacing with uniform distribution.")
        topk_probs = np.ones(len(topk_probs)) / len(topk_probs)  # Uniform distribution

    word = np.random.choice(topk_idx, p = topk_probs, size=1)[0]
    if flag:
        print("topk_idx:", topk_idx)
        print("token:", word)
    if word >= len(probs):
        print("word is out of range, replacing with the last token.")
        return len(probs) - 1
    return word
    
def test(opt, n_target_bar = 32, output_folder_dir = '', midi_path = '', prompt = None):
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
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        print('Start generating')
        while current_generated_bar < n_target_bar:
            print("\r", current_generated_bar, end="")
            # input
            if initial_flag:
                temp_x = np.zeros((batch_size, original_length))
                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                # truncate the sequence if it is too long
                initial_length = temp_x.shape[1]
                if initial_length >= (opt.max_input_length - 1):
                    temp_x = temp_x[:, -opt.max_input_length+1:]

                initial_flag = 0
                
            else:
                current_length = temp_x.shape[1]
                # Case 1: Sequence length has not reached max_position_size
                if current_length < (opt.max_input_length - 2):
                    # Append new token(s) without shifting
                    temp_x_new = np.zeros((batch_size, 1))
                    for b in range(batch_size):
                        temp_x_new[b][0] = words[b][-1]

                    # Append the new column
                    temp_x = np.concatenate([temp_x, temp_x_new], axis=1)
                
                # Case 2: Sequence length has reached max_position_size
                else:
                    # Shift the sequence to the left by one position
                    temp_x = np.roll(temp_x, shift=-1, axis=1)
                    
                    # Insert new token(s) at the last position
                    for b in range(batch_size):
                        temp_x[b, -1] = words[b][-1]
            
            temp_x = torch.Tensor(temp_x).long()

            output_logits = model(temp_x.to(opt.device))
            
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
                words=words[0][initial_length:],
                word2event=word2event,
                output_path=os.path.join(opt.output_folder, midi_path),
                prompt_path=None)
        else:
            utils.write_midi(
                words=words[0],
                word2event=word2event,
                output_path=os.path.join(opt.output_folder, midi_path),
                prompt_path=None)
    
    

def main():
    # for prompt in glob.glob('./prompt_song/*.mid'):
    for i in range(20):
        # midi_path=f'{prompt.split("/")[-1].split(".")[0]}_generated.mid'
        midi_path = f'sample_{i}.mid'
        test(
            opt,
            n_target_bar=32, 
            output_folder_dir=opt.output_folder,
            midi_path=midi_path,
            prompt=None)

if __name__ == '__main__':
    main()

