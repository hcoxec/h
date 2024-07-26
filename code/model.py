import torch
import math
import attr
import json

from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from functools import partial
from torch.optim import Adam
from collections import defaultdict
from datasets import load_dataset
from itertools import tee
from datetime import datetime
from datetime import date as D
from os import makedirs

from estimator import Information

@attr.s
class Vocab:
    '''
    An an object for storing the vocab and mapping
    between tokens and indicies
    '''
    tokens = attr.ib()
    token2index = attr.ib()
    index2token = attr.ib()
    
    def __len__(self):
        return len(self.tokens)+1
    
    def __getitem__(self, item):
        if type(item) == str:
            return self.token2index[item]
        elif type(item) == int:
            return self.index2token[item]
        
def preprocess(dataset):
    vocab = ['<PAD>', '<BOS>', '<EOS>', '<CLS>']
    for example in dataset:
        for token in example['question'].split():
            if token not in vocab:
                vocab.append(token)
                
        for token in example['query'].split():
            if token not in vocab:
                vocab.append(token)
            
    token2index, index2token = {}, {}
    
    for i, token in enumerate(vocab):
        token2index[token] = i
        index2token[i] = token
        
    return vocab, token2index, index2token
    
    
def get_example(batch, dataset='cfq'):
    if dataset == 'cfq':
        return [i['question'].split() for i in batch], [i['query'].split() for i in batch]

def get_batch(batch, token2index):
    input_ids, output_ids = [], []
    inputs, outputs = get_example(batch)
    for example in inputs:
        sequence = ['<CLS>']
        sequence.extend(example)
        input_ids.append(
            torch.tensor([token2index[t] for t in sequence])
        )
    for example in outputs:
        sequence = ['<BOS>']
        sequence.extend(example)
        sequence.append('<EOS>')
        output_ids.append(
            torch.tensor([token2index[t] for t in sequence])
        )
    return input_ids, output_ids

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    


class EncoderDecoder(nn.Module):
    "Builds a Seq2Seq Transformer"

    def __init__(
        self, vocab, d_model=128, n_heads=4, 
        device='cpu', dropout=0.1, max_len=1000, pad_idx=0,
        log_file='logs/h_test', h_estimator=None,
    ):
        super(EncoderDecoder, self).__init__()
        
        self.vocab = vocab
        self.device = device
        self.vocab_size = len(vocab)
        self.pad_idx = pad_idx
        self.trace=0
        
        if h_estimator is None:
            self.h_estimator = Information(
                n_bins=20, 
                n_heads=128, 
                dist_fn='unit_cube', 
                smoothing_fn='discrete'
            )
        else: self.h_estimator = h_estimator
        
        date = D.today()
        self.log_path = f"{log_file}/{date.month}_{date.day}/{datetime.now().strftime('%H_%M_%S')}"
        
        makedirs(self.log_path, exist_ok=True)
        
        self.log_params = {
            'd_model':d_model,
            'n_heads':n_heads,
            'dropout':dropout,
        }
        self.init_log()
        
        
        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=d_model, padding_idx=pad_idx, device=device)
        self.dec_embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=d_model, padding_idx=pad_idx, device=device)

        self.position = PositionalEncoding(d_model, dropout, max_len).to(device)
        self.model = torch.nn.Transformer(
            d_model=d_model, nhead=n_heads, num_encoder_layers=2, num_decoder_layers=1, 
            dim_feedforward=d_model*4, dropout=0.1, layer_norm_eps=1e-05, batch_first=True, 
            norm_first=False, device=device, dtype=None
        )
        self.final_linear = nn.Linear(d_model, len(vocab), device=device)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def save(self, filename='model'):
        torch.save(self.state_dict(), f'{self.log_path}/{filename}_checkpoint.md5')
        
    def accuracy(self, logits, output_mask, out_decoder, return_scores=False):
        exact_acc, partial_acc = [], []
        for i, seq in enumerate(logits.argmax(dim=-1)):
            prediction = seq[output_mask[i]]
            score = out_decoder[i].to(prediction.device).eq(prediction).float().mean()
            exact_acc.append(score if score==1.0 else 0.0)
            partial_acc.append(score)
            
        exact_acc, partial_acc = torch.tensor(exact_acc), torch.tensor(partial_acc)
        
        if return_scores:
            return exact_acc, partial_acc

        return exact_acc.mean(), partial_acc.mean()
    
    def decoder_tokens(self, outputs):
        in_decoder, out_decoder = [], []
        for ex in outputs:
            in_decoder.append(ex[:-1])
            out_decoder.append(ex[1:])
            
        return in_decoder, out_decoder
    
    def pad(self, tensor):
        return  nn.utils.rnn.pad_sequence(tensor, batch_first=True).to(self.device)
            
    def forward(self, batch, compute_loss=True, compute_acc=True, estimate_h=True):
     
        inputs, in_decoder, out_decoder, src_pad_mask, tgt_pad_mask, output_mask = self.prep_batch(batch)
        decoder_mask = self.model.generate_square_subsequent_mask(tgt_pad_mask.shape[-1], device=self.device).lt(0)
        
        x, y = self.embedding(inputs), self.dec_embedding(in_decoder)
        x, y = self.position(x), self.position(y)
        
        encodings = self.model.encoder(x,src_key_padding_mask=src_pad_mask)
        if self.trace == 0:
            self.write_log({'encs':encodings[0].tolist()})
            self.trace +=1
            
        x = self.model.decoder(
            y, encodings, 
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=decoder_mask
        )

        logits = F.log_softmax(self.final_linear(x), dim=-1)
        
        to_return = {}
        if compute_acc:
            exact_acc, partial_acc = self.accuracy(logits, output_mask, out_decoder)
            to_return['exact_acc_train'] = exact_acc
            to_return['partial_acc_train'] = partial_acc
        
        if compute_loss:
            loss = F.cross_entropy(logits[output_mask], torch.cat(out_decoder).to(logits.device))
            to_return['loss_train'] = loss
            
        if estimate_h:
            h = self.h_estimate(encodings[inputs.ne(0)], batch[0])
            to_return.update(h)
            
        self.log(to_return)
            
        return to_return
    
    def evaluate(self, batch):
        self.train(mode=False)
        x = self.forward(batch)
        self.train()
        return x
    
    def h_estimate(self, encoder_states, idx_input):
        label_ids = get_token_labels(idx_input)
        self.h_estimator.reset(keep_bins=False)
        self.h_estimator.batch_count(encoder_states, label_ids)
        return self.h_estimator.analyse()
        
        
    def log(self, data:dict):
        self.log_state.update(data)
    
    def write_log(self, hyperparams={}, file='train'):
        self.log_state.update(hyperparams)
        
        to_write = {'mode':file}
        for item in self.log_state:
            if (type(self.log_state[item]) == torch.Tensor):
                to_write[item] = self.log_state[item].item()
            else:
                to_write[item] = self.log_state[item]
        
                
        with open(f"{self.log_path}/{file}.json", "a") as outfile:
            outfile.write(json.dumps(to_write)+'\n')
            
        self.init_log()
            
    def init_log(self, hyperparams={}):
        self.log_state = {}
        
        self.log_state.update(self.log_params)
        self.log_state.update(hyperparams)
    
    def prep_batch(self, batch):
        inputs, outputs = batch
        
        in_decoder, out_decoder = self.decoder_tokens(outputs)
        inputs, in_decoder = self.pad(inputs), self.pad(in_decoder)
        
        pad_idx = self.vocab['<PAD>']
        src_pad_mask, tgt_pad_mask, output_mask = inputs.eq(pad_idx), in_decoder.eq(pad_idx), in_decoder.gt(pad_idx)
        
        return inputs, in_decoder, out_decoder, src_pad_mask, tgt_pad_mask, output_mask
    
    def mask_and_pad(self, indices):
        pad_idx = self.vocab['<PAD>']
        indices = self.pad(indices)
        return indices, indices.eq(pad_idx)
        
    def encode(self, batch):
        inputs, outputs = batch
        
        inputs, input_mask = self.mask_and_pad(inputs)
        x = self.position(self.embedding(inputs))
        
        return self.model.encoder(x, src_key_padding_mask=input_mask), inputs
    
    def get_encodings(self, dataloader):
        encoder_states, idx_inputs = [], []
        self.train(False)
        with torch.inference_mode():
            for b in dataloader:
                encodings, inputs = self.encode(b)
                encoder_states.append(encodings[inputs.ne(0)])
                idx_inputs.extend(b[0])
                
        self.train(True)
    
        return torch.cat(encoder_states), idx_inputs
    
    def greedy_decode(self, batch):
        inputs, in_decoder, out_decoder, src_pad_mask, tgt_pad_mask, output_mask = self.prep_batch(batch)
        
        bos_idx, eos_idx = self.vocab['<BOS>'], self.vocab['<EOS>']
        bs, max_len = in_decoder.shape[0], in_decoder.shape[-1]
        
        self.train(False)
        with torch.inference_mode():
            enc = self.position(self.embedding(inputs))
            
            enc = self.model.encoder(enc,src_key_padding_mask=src_pad_mask)
            
            in_decoder = (bos_idx*torch.ones([bs, 1]).to(self.device)).int()
            tgt_pad_mask = torch.zeros([bs, 1]).to(self.device).bool()
            
            done_decoding = torch.zeros([bs, 1]).to(self.device)
            
            for step in range(max_len):
                
                incremental_mask = self.model.generate_square_subsequent_mask(tgt_pad_mask.shape[-1], device=self.device).lt(0)
                
                y = self.position(self.dec_embedding(in_decoder))
                y_hat = self.model.decoder(
                    y,
                    enc, 
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_pad_mask,
                    tgt_mask=incremental_mask
                )

                logits = F.log_softmax(self.final_linear(y_hat), dim=-1)
                predictions = logits.argmax(dim=-1)[:,-1].unsqueeze(-1)
                
                is_eos = predictions.eq(eos_idx)
                done_decoding += is_eos
                
                in_decoder = torch.cat([in_decoder, predictions], dim=-1)
                tgt_pad_mask = torch.cat([tgt_pad_mask, done_decoding.gt(0)], dim=-1).bool()
            
        self.train(True)
        exact_acc, partial_acc = self.accuracy(logits, output_mask, out_decoder, return_scores=True)
        
        return F.cross_entropy(logits[output_mask], torch.cat(out_decoder).to(logits.device)), exact_acc, partial_acc
    
    def auto_regressive_eval(self, eval_loader):
        ea, pa, ls = [], [], []
        for b in eval_loader:
            loss, exact_acc, partial_acc = self.greedy_decode(b)
            ea.append(exact_acc)
            pa.append(partial_acc)
            ls.append(loss)
                
        return torch.cat(ea).mean().item(), torch.cat(pa).mean().item(), torch.stack(ls).mean().item()


def n_gram(sequence, n):
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
            
    return list(zip(*iterables))  # Unpack and flattens the iterables.

def idx_ngrams(max_len, n):
    idxs = n_gram(range(max_len), n=n)
    overlap_matrix = []
    for i in range(max_len):
        ii = []
        for inn, ngr in enumerate(idxs):
            if i in ngr:
                ii.append(inn)
        overlap_matrix.append(ii)
        
    return overlap_matrix

def get_token_labels(
    examples,
    seq_labels=None,
    labels=['token','bigram','trigram'],
    pos_dict=None,
    shift_index=False,
    pos_labels=None,
    dep_labels=None,
):
    token_ids = defaultdict(lambda: [])
    pos_ids = defaultdict(lambda: [])
    dep_ids = defaultdict(lambda: [])
    general_label_ids = defaultdict(lambda: [])
    bigram_ids = defaultdict(lambda: defaultdict(lambda: []))
    trigram_ids = defaultdict(lambda: defaultdict(lambda: []))
    bow_ids = defaultdict(lambda: defaultdict(lambda: []))
    token_info = []
    
    token_vectors = []
    
    i = 0 #assign an id to each token instance in the dataset
    for i_e, example in enumerate(examples):
        tokens = example if type(example) == list else example.tolist()
        bigrams, trigrams = n_gram(tokens, n=2), n_gram(tokens, n=3)
        bigram_idxs, trigram_idxs = idx_ngrams(len(tokens), n=2), idx_ngrams(len(tokens), n=3)
        if shift_index:
            tokens = tokens[1:]
        for i_t, token in enumerate(tokens):
            
            #token = token.lower() if uncased else token
            token_ids[token].append(i)
            bis, tris = [], []
            
            if ('pos' in labels) and (pos_dict is not None):
                token_pos = pos_dict[token]
                pos_ids[token_pos].append(i)
                
            if ('pos' in labels) and (pos_labels is not None):
                pos_ids[pos_labels[i_e][i_t]].append(i)
            
            if ('dep' in labels) and (dep_labels is not None):
                dep_ids[dep_labels[i_e][i_t]].append(i)
                
            if ('label' in labels) and (seq_labels is not None):
                general_label_ids[seq_labels[i_e][i_t]].append(i)
            
                
            if 'bigram' in labels:
                for bi in bigram_idxs[i_t]:
                    bigram_ids[token][bigrams[bi]].append(i)
                    bis.append(bigrams[bi])
            
            if 'trigram' in labels:
                for tri in trigram_idxs[i_t]:
                    trigram_ids[token][trigrams[tri]].append(i)
                    tris.append(trigrams[tri])
            
            if 'bow' in labels:
                for i_tt, other_token in enumerate(tokens):
                    if i_t != i_tt:
                        bow_ids[token][other_token].append(i)
                            
            i+=1
            
    ids = {
        'token': token_ids,
        'pos':pos_ids,
        'dep':dep_ids,
        'bigram':bigram_ids,
        'trigram':trigram_ids,
        'bow':bow_ids,
        'label':general_label_ids,
    }
    
    label_ids = {}
    for id_type in labels:
        label_ids[id_type] = ids[id_type]
    
    return label_ids