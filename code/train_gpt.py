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
    parser.add_argument('--n_epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--max_input_length', type=int, help='max input length', default=1024)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension', default=512)
    parser.add_argument('--embed_dim', type=int, help='embedding dimension', default=512)
    parser.add_argument('--n_head', type=int, help='number of heads', default=8)
    parser.add_argument('--n_layer', type=int, help='number of layers', default=12)
    parser.add_argument('--dropout', type=float, help='dropout rate', default=0.1)
    parser.add_argument('--group_size', type=int, help='number of group segments', default=5)
    parser.add_argument('--device', type=str, help='device to use', default='cuda:0')
    parser.add_argument('--dict_path', type=str, help='the dictionary path.', default='./dictionary.pkl')
    parser.add_argument('--ckpt_dir', type=str, help='the checkpoint path.', default='')
    parser.add_argument('--ckpt_folder', type=str, help='the checkpoint folder path.', default='./checkpoints')
    args = parser.parse_args()
    return args

opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
## set the input length. must be same with the model config
X_LEN = opt.max_input_length

class NewsDataset(Dataset):
    def __init__(self, midi_l = [], prompt = ''):
        self.midi_l = midi_l
        self.x_len = X_LEN
        self.dictionary_path = opt.dict_path
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data(self.midi_l)
    
    def __len__(self):
        return len(self.parser)  
    
    def __getitem__(self, index):
        return self.parser[index]
    
    def chord_extract(self, midi_path, max_time):
        ####################################################
        # add your chord extraction method here if you want
        ####################################################
        return
    
    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end

        # if you use chord items, you need to add chord_items into "items"
        # e.g. items = tempo_items + note_items + chord_items
        items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
        
    def prepare_data(self, midi_paths):
        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)
        # event to word
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        # something is wrong
                        # you should handle it for your own purpose
                        print('something is wrong! {}'.format(e))
            all_words.append(words)
        
        # all_words is a list containing words list of all midi files
        # all_words = [[tokens of midi], [tokens of midi], ...]

        # you can cut the data into what you want to feed into model
        # Warning : this example cannot use in transformer_XL, you must implement group segments by yourself
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words)-self.x_len-1, self.x_len):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                pairs.append([x, y])
            # abandon last segments in a midi
            pairs = pairs[0:len(pairs)-(len(pairs)%5)]
            segments = segments + pairs
        segments = np.array(segments)
        print(segments.shape)
        return segments

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        # Construct the configuration for the Transformer-XL model
        self.config = GPT2Config(
            vocab_size=len(event2word),
            n_positions=opt.max_input_length,
            n_embd=opt.embed_dim,
            n_head=opt.n_head,
            n_layer=opt.n_layer,
        )
        # Load the GPT-2 model
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
    word = np.random.choice(topk_idx, p = topk_probs)
    return word
       
def train(opt, is_continue = False):
    epochs = opt.n_epochs
    lr = opt.lr
    checkpoints_path = opt.ckpt_dir

    # create data list
    # use glob to get all midi file path
    train_list = glob.glob('dataset/*.mid')
    print('train list len =', len(train_list))

    # dataset
    train_dataset = NewsDataset(train_list)
    # dataloader
    BATCH_SIZE = opt.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    print('Dataloader is created')

    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device(opt.device)
    else:
        device = torch.device("cpu")
    
    # create model
    if not is_continue:
        start_epoch = 1
        model = Model(opt).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    else:
        # wheather checkpoint_path is exist
        if os.path.isfile(checkpoints_path):
            checkpoint = torch.load(checkpoints_path)
        else:
            os._exit()
        start_epoch = checkpoint['epoch'] + 1

        model = Model(opt).to(device)
        model.load_state_dict(checkpoint['model'])

        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Model is created \nStart training')
    
    model.train()
    losses = []
    try:
        os.makedirs(opt.ckpt_folder, exist_ok=True)
        print("dir is created")
    except:
        pass
    
    for epoch in range(start_epoch, epochs+1):
        single_epoch = []
        for i, batch in tqdm(enumerate(train_dataloader), desc='Batches', unit='batch', total=len(train_dataloader)):
            x = batch[:, 0, :].to(device).long()
            y = batch[:, 1, :].to(device).long()
            output_logit = model(x)
            loss = nn.CrossEntropyLoss()(output_logit.permute(0,2,1), y)
            loss.backward()
            single_epoch.append(loss.to('cpu').mean().item())
            optimizer.step()
            optimizer.zero_grad()

        single_epoch = np.array(single_epoch)
        losses.append(single_epoch.mean())
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch,losses[-1]))
        if epoch % 10 == 0 or epoch == 1:
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': losses[-1],
                        }, os.path.join(opt.ckpt_folder, 'epoch_%03d.pkl'%epoch))
        if losses[-1] <= 0.1:
            break
    np.save(os.path.join(opt.ckpt_folder, 'training_loss'), np.array(losses))

def main():
    train(opt, is_continue = False)

if __name__ == '__main__':
    main()

