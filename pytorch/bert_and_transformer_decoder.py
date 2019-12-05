import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchtext
from torchtext import data, datasets
from keras.preprocessing.sequence import pad_sequences
from transformers import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

# Examples of pre-trained Transformer-based models that are available
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base')]

class BERTEncoder(nn.Module):
    def __init__(self):
        super(BERTEncoder, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, x, mask):
        # NOTE: torch.no_grad() means the BERT params are frozen
        with torch.no_grad():
            # Not using the mask here for now ...
            output = self.model(x)
        return output

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=768, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    encoder = BERTEncoder()
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    # NOTE: Only use xavier initialisation for decoder params. Don't touch the BERT params!
    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    model = EncoderDecoder(
        encoder,
        decoder,
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    return model

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        batch.src = batch.src.to(DEVICE)
        batch.trg = batch.trg.to(DEVICE)
        batch.src_mask = batch.src_mask.to(DEVICE)
        batch.trg_mask = batch.trg_mask.to(DEVICE)
        batch.trg_y = batch.trg_y.to(DEVICE)
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.sum() and len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

def preprocess_data(**kwargs):
    BOS_WORD = kwargs['start_token']
    EOS_WORD = kwargs['end_token']
    BLANK_WORD = kwargs['pad_token']
    MAX_LEN = 100

    def tokenize_src(text):
        if kwargs['transformer'] == 'bert':
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            result = bert_tokenizer.encode(text, add_special_tokens=True)
            result = pad_sequences([result], maxlen=100, dtype="long", truncating="post", padding="post")[0].tolist()
            # print("Tokenization result %s" % (result))
            return result
        else:
            spacy_en = spacy.load('en')
            return [tok.text for tok in spacy_en.tokenizer(text)]

    def tokenize_tgt(text):
        if kwargs['transformer'] == 'bert':
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            result = bert_tokenizer.encode(text, add_special_tokens=True)
            result = pad_sequences([result], maxlen=100, dtype="long", truncating="post", padding="post")[0].tolist()
            # print("Tokenization result %s" % (result))
            return result
        else:
            spacy_en = spacy.load('en')
            return [tok.text for tok in spacy_en.tokenizer(text)]
    
    SRC = torchtext.data.Field(tokenize=tokenize_src, fix_length=MAX_LEN, pad_token=BLANK_WORD)
    TGT = torchtext.data.Field(tokenize=tokenize_tgt, fix_length=MAX_LEN, pad_token=BLANK_WORD)

    train, val, test = torchtext.data.TabularDataset.splits(
        path='./data/geo', train='geo880_train600.tsv',
        validation='geo880_dev100.tsv', test='geo880_test280.tsv',
        format='tsv', fields=[('src', SRC), ('trg', TGT)])

    SRC.build_vocab(train.src)
    TGT.build_vocab(train.trg)

    return SRC, TGT, train, val, test

# SRC, TGT, train, val, test = preprocess_data(start_token='<s>', end_token='</s>', pad_token='blank', transformer=vanilla)
SRC, TGT, train, val, test = preprocess_data(start_token='[CLS]', end_token='[SEP]', pad_token='[PAD]', transformer='bert')

# def load_data_for_bert():
#     train_df = pd.read_csv("data/geo/geo880_train600.tsv",
#                             delimiter='\t',
#                             header=None,
#                             names=['nat_lang', 'logical_lang'])
#     valid_df = pd.read_csv("data/geo/geo880_dev100.tsv",
#                             delimiter='\t',
#                             header=None,
#                             names=['nat_lang', 'logical_lang'])
#     test_df = pd.read_csv("data/geo/geo880_test280.tsv",
#                             delimiter='\t',
#                             header=None,
#                             names=['nat_lang', 'logical_lang'])
#     return train_df, valid_df, test_df

# def preprocess_data_for_bert(train_df, valid_df, test_df):
#      train_data = {'src': train_df.nat_lang.values.tolist(), 'trg': train_df.nat_lang.values.tolist()}
#      valid_data = {'src': valid_df.nat_lang.values.tolist(), 'trg': valid_df.nat_lang.values.tolist()}
#      test_data = {'src': test_df.nat_lang.values.tolist(), 'trg': test_df.nat_lang.values.tolist()}

#      bert_model = BertModel.from_pretrained('bert-base-uncased')
#      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#      # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
#      # BERT requires a fixed-length, so pad every sentence to a max length of 120 (found by looking at the geo data)
#      # Convert all sentences into the tensor required by BERT
#      train_input_ids = [bert_tokenizer.encode(x, add_special_tokens=True) for x in train_data['src']]
#      train_input_ids = pad_sequences(train_input_ids, maxlen=120, dtype="long", truncating="post", padding="post")
#      train_input_ids = torch.tensor(train_input_ids)
     
#      valid_input_ids = [bert_tokenizer.encode(x, add_special_tokens=True) for x in valid_data['src']]
#      valid_input_ids = pad_sequences(valid_input_ids, maxlen=120, dtype="long", truncating="post", padding="post")
#      valid_input_ids = torch.tensor(valid_input_ids)
     
#      test_input_ids = [bert_tokenizer.encode(x, add_special_tokens=True) for x in test_data['src']]
#      test_input_ids = pad_sequences(test_input_ids, maxlen=120, dtype="long", truncating="post", padding="post")
#      test_input_ids = torch.tensor(test_input_ids)

# train_df, valid_df, test_df = load_data_for_bert()
# preprocess_data_for_bert(train_df, valid_df, test_df)

class MyIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(data_set, random_shuffler):
                for og_batch in torchtext.data.batch(data_set, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(og_batch, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


if True:
    pad_idx = TGT.vocab.stoi["[PAD]"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    # model.cuda()
    model = model.to(DEVICE)
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    # criterion.cuda()
    criterion = criterion.to(DEVICE)
    BATCH_SIZE = 1000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
for epoch in range(10):
    print("Starting epoch %s ..." % (epoch))
    model.train()
    run_epoch((rebatch(pad_idx, b) for b in train_iter), 
              model, 
              SimpleLossCompute(model.generator, criterion, opt=model_opt))
    model.eval()
    loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                      model, 
                      SimpleLossCompute(model.generator, criterion, None))
    print(loss)
    print("Ending epoch %s ..." % (epoch))
    torch.save(model.state_dict(), '/checkpoints/bert_and_transformer_decoder%s.pt' % (epoch))

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    out = greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break

