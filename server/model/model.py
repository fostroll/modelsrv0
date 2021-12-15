from junky.dataset import BertDataset
import os
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .model_head import ModelHead


class Model(nn.Module):

    def __init__(self, *args):
        super().__init__()

        if len(args) == 1:
            # path
            emb, tokenizer, head = self._load(args[0])
        elif len(args) == 2:
            # bert_path, model_head
            emb, tokenizer = self._load_bert(args[0])
            head = args[1]
        elif len(args) == 3:
            # emb_model, tokenizer, model_head
            emb, tokenizer, head = args
        else:
            raise ValueError('ERROR: Invalid agruments.')
            
        self.emb_model = emb
        self.tokenizer = tokenizer
        self.model_head = head

        self.ds = BertDataset(emb, tokenizer)

    _model_head_dn = 'model_head'
    _model_emb_dn = 'model_emb'

    @staticmethod
    def _save_bert(emb_model, tokenizer, path):
        tokenizer.save_pretrained(path)
        emb_model.save_pretrained(path)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self._save_bert(self.emb_model, self.tokenizer,
                        os.path.join(path, self.__class__._model_emb_dn))
        self.model_head.save(os.path.join(path, self.__class__._model_head_dn))

    @classmethod
    def _load_bert(cls, path):
        tokenizer = AutoTokenizer.from_pretrained(path, do_lower_case=False)
        config = AutoConfig.from_pretrained(path, output_hidden_states=True,
                                            output_attentions=False)
        model = AutoModel.from_pretrained(path, config=config)
        model.eval()
        return model, tokenizer

    @classmethod
    def _load(cls, path):
        emb, tokenizer = cls._load_bert(os.path.join(path, cls._model_emb_dn))
        head = ModelHead.load(os.path.join(path, cls._model_head_dn))
        return emb, tokenizer, head

    @classmethod
    def load(cls, path):
        emb, tokenizer, head = cls._load(path)
        model = cls(emb, tokenizer, head)
        model.eval()
        return model

    def forward(self, sentences, *args, labels=None):
        x = self.ds.transform(sentences, max_len=0, batch_size=len(sentences),
                              hidden_ids=10, aggregate_hiddens_op='cat',
                              aggregate_subtokens_op='absmax', with_grad=self.training,
                              save=False, loglevel=0)
        x, x_lens = self.ds._collate(x, with_lens=True)
        return self.model_head(x, x_lens, *args, labels=labels)
