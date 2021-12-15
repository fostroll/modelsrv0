from collections import Iterable, OrderedDict
import junky
from junky import BaseConfig, to_device
from junky.layers import CharEmbeddingRNN, CharEmbeddingCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os


class ModelHeadConfig(BaseConfig):
    vec_emb_dim = None
    alphabet_size = 0
    char_pad_idx = 0
    rnn_emb_dim = None
    cnn_emb_dim = None
    cnn_kernels = [1, 2, 3, 4, 5, 6]
    tag_emb_params = None
    emb_bn = True
    emb_do = .2
    final_emb_dim = 512
    pre_bn=True
    pre_do=.5
    lstm_layers = 1
    lstm_do = 0
    post_bn = True
    post_do = .4

    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels


class BatchNorm(nn.BatchNorm1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ModelHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if isinstance(config.cnn_kernels, Iterable):
            config.cnn_kernels = list(config.cnn_kernels)

        assert config.lstm_layers >= 1, \
            'ERROR: `lstm_layers` must not be lower than `1`.'
        assert config.final_emb_dim % 2 == 0, \
            'ERROR: `final_emb_dim` must be even ' \
           f"(now it's `{final_emb_dim}`)."

        if config.vec_emb_dim is None:
            config.vec_emb_dim = 0

        if config.rnn_emb_dim:
            self.rnn_emb_l = \
                CharEmbeddingRNN(alphabet_size=config.alphabet_size,
                                 emb_dim=config.rnn_emb_dim,
                                 pad_idx=config.char_pad_idx)
        else:
            self.rnn_emb_l = None
            config.rnn_emb_dim = 0

        if config.cnn_emb_dim:
            self.cnn_emb_l = \
                CharEmbeddingCNN(alphabet_size=config.alphabet_size,
                                 emb_dim=config.cnn_emb_dim,
                                 pad_idx=config.char_pad_idx,
                                 kernels=config.cnn_kernels)
        else:
            self.cnn_emb_l = None
            config.cnn_emb_dim = 0

        self.tag_emb_l = self.tag_emb_ls = None
        tag_emb_dim = 0
        if config.tag_emb_params:
            if isinstance(config.tag_emb_params, dict):
                tag_emb_dim = config.tag_emb_params['dim'] or 0
                if tag_emb_dim:
                    self.tag_emb_l = \
                        nn.Embedding(config.tag_emb_params['num'], tag_emb_dim,
                                     padding_idx=config.tag_emb_params['pad_idx'])
            else:
                self.tag_emb_ls = nn.ModuleList()
                for emb_params in config.tag_emb_params:
                    tag_emb_dim_ = emb_params['dim']
                    if tag_emb_dim_:
                        tag_emb_dim += tag_emb_dim_
                        self.tag_emb_ls.append(
                            nn.Embedding(emb_params['num'], tag_emb_dim_,
                                         padding_idx=emb_params['pad_idx'])
                        )
                    else:
                        self.tag_emb_ls.append(None)

        joint_emb_dim = config.vec_emb_dim + config.rnn_emb_dim + \
                        config.cnn_emb_dim + tag_emb_dim
        assert joint_emb_dim, \
            'ERROR: At least one of `config.*_emb_dim` must be specified'

        # PREPROCESS #########################################################
        modules = OrderedDict()
        if config.emb_bn:
            modules['emb_bn'] = BatchNorm(joint_emb_dim)
        if config.emb_do:
            modules['emb_do'] = nn.Dropout(p=config.emb_do)

        layers = []
        def add_layers(dim, new_dim):
            ls = []
            ls.append(('pre_fc{}_l',
                       nn.Linear(in_features=new_dim, out_features=dim)))
            if config.pre_bn:
                ls.append(('pre_bn{}', BatchNorm(dim)))
            ls.append(('pre_nl{}', nn.ReLU()))
            if config.pre_do:
                ls.append(('pre_do{}', nn.Dropout(p=config.pre_do)))
            layers.append(ls)

        dim = config.final_emb_dim
        while joint_emb_dim / dim > 2:
            new_dim = int(dim * 1.5)
            add_layers(dim, new_dim)
            dim = new_dim
        add_layers(dim, joint_emb_dim)
        for idx, layer in enumerate(reversed(layers)):
            for name, module in layer:
                modules[name.format(idx)] = module

        self.pre_seq_l = nn.Sequential(modules)
        ######################################################################

        self.lstm_l = nn.LSTM(
            input_size=config.final_emb_dim,
            hidden_size=config.final_emb_dim // 2,
            num_layers=config.lstm_layers, batch_first=True,
            dropout=config.lstm_do, bidirectional=True
        )
        self.T = nn.Linear(config.final_emb_dim, config.final_emb_dim)
        nn.init.constant_(self.T.bias, -1)

        # POSTPROCESS ########################################################
        modules = OrderedDict()
        if config.post_bn:
            modules['post_bn'] = nn.BatchNorm1d(config.final_emb_dim)
        if config.post_do:
            modules['post_do'] = nn.Dropout(p=config.post_do)

        modules['out_fc_l'] = nn.Linear(in_features=config.final_emb_dim,
                                        out_features=config.num_labels)
        self.post_seq_l = nn.Sequential(modules)
        ######################################################################

#         self.criterion = nn.BCEWithLogitsLoss(
#             reduction='mean',
#             #reduction='none',
#             #reduction='sum',
#             #pos_weight=None
#             pos_weight=torch.tensor(6)
#         )

    _model_fn = 'model.pt'
    _model_config_fn = 'model_config.json'

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.config.save(os.path.join(path, self.__class__._model_config_fn))
        torch.save(self.state_dict(),
                   os.path.join(path, self.__class__._model_fn))

    @classmethod
    def load(cls, path):
        config = ModelHeadConfig.load(os.path.join(path, cls._model_config_fn))
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(path, cls._model_fn),
                                         map_location=junky.CPU))
        model.eval()
        return model

    def forward(self, x, x_lens, x_ch, x_ch_lens, *x_t, labels=None):
        """
        x:         [N (batch), L (sentences), C (words + padding)]
        lens:      number of words in sentences
        x_ch:      [N, L, C (words + padding), S (characters + padding)]
        x_ch_lens: [L, number of characters in words]
        *x_t:      [N, L, C (upos indices)], [N, L, C (xpos indices)], ...
        labels:    [N, L, C]

        Returns logits if label is `None`, (logits, loss) otherwise.
        """
        config = self.config
        device = next(self.parameters()).device

        x_ = []
        if config.vec_emb_dim:
            assert x.shape[2] == config.vec_emb_dim, \
                  f'ERROR: Invalid vector size: `{x.shape[2]}` ' \
                  f'whereas `vec_emb_dim={config.vec_emb_dim}`'
            x_.append(to_device(x, device))
        if self.rnn_emb_l:
            x_.append(self.rnn_emb_l(to_device(x_ch, device), x_ch_lens))
        if self.cnn_emb_l:
            x_.append(self.cnn_emb_l(to_device(x_ch, device),
                                      to_device(x_ch_lens, device)))
        if self.tag_emb_l:
            x_.append(self.tag_emb_l(to_device(x_t[0], device)))
        elif self.tag_emb_ls:
            for l_, x_t_ in zip(self.tag_emb_ls, to_device(x_t, device)):
                if l_:
                    x_.append(l_(x_t_))

        x = x_[0] if len(x_) == 1 else torch.cat(x_, dim=-1)

        x = self.pre_seq_l(x)

        x_ = pack_padded_sequence(x, x_lens.cpu(), batch_first=True,
                                  enforce_sorted=False)
        output, (h_n, c_n) = self.lstm_l(x_)
        # h_n.shape => [batch size, num layers * num directions, hidden size]

        ## 1. if we want to use h_n:
        x_ = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) \
                 if self.lstm_l.bidirectional else \
             h_n[-1, :, :]
        # x_.shape => [batch size, hidden size]
        x = x_
        #####

        '''
        ## 2. if we want to use output:
        x_, _ = pad_packed_sequence(output, batch_first=True)

        gate = torch.sigmoid(self._T(x))
        x = x_ * gate + x * (1 - gate)

        x = x_[:, -1, :]
        # or
        x = torch.max(x_, dim=1).values
        # or
        x = torch.mean(x_, dim=1)
        # or
        x = torch.sum(x_, dim=1)
        #####
        '''

        logits = self.post_seq_l(x)
        if not self.training:
            logits.sigmoid_()

        if labels is not None:
            loss = \
                F.binary_cross_entropy_with_logits(logits, labels.float(),
                                                   reduction='mean',
                                                   pos_weight=pos_weights) \
                    if self.training else \
                F.binary_cross_entropy(logits, labels.float(), reduction='mean')
            logits = logits, loss

        return logits
