from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F

MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base')]

class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def forward(self, x):
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        input_ids = torch.tensor([self.tokenizer.encode(x, add_special_tokens=True)])
        with torch.no_grad():
            output = self.model(input_ids)
        import ipdb; ipdb.set_trace()
        return output

class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()
    
    def forward(self, x):
        

encoder = Encoder()
print(encoder("Derp"))
        
