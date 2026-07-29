# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin

NISQA Inference Module - Simplified interface for speech quality prediction
"""
import os
import copy
import math

import librosa as lb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


#%% Models
class NISQA_DIM(nn.Module):
    '''
    NISQA_DIM: The main speech quality model with speech quality dimension
    estimation (MOS, Noisiness, Coloration, Discontinuity, and Loudness).
    The module loads the submodules for framewise modelling (e.g. CNN),
    time-dependency modelling (e.g. Self-Attention or LSTM), and pooling
    (e.g. max-pooling or attention-pooling)
    '''
    def __init__(self,
            ms_seg_length=15,
            ms_n_mels=48,

            cnn_model='adapt',
            cnn_c_out_1=16,
            cnn_c_out_2=32,
            cnn_c_out_3=64,
            cnn_kernel_size=3,
            cnn_dropout=0.2,
            cnn_pool_1=[24,7],
            cnn_pool_2=[12,5],
            cnn_pool_3=[6,3],
            cnn_fc_out_h=None,

            td='self_att',
            td_sa_d_model=64,
            td_sa_nhead=1,
            td_sa_pos_enc=None,
            td_sa_num_layers=2,
            td_sa_h=64,
            td_sa_dropout=0.1,
            td_lstm_h=128,
            td_lstm_num_layers=1,
            td_lstm_dropout=0,
            td_lstm_bidirectional=True,

            td_2='skip',
            td_2_sa_d_model=None,
            td_2_sa_nhead=None,
            td_2_sa_pos_enc=None,
            td_2_sa_num_layers=None,
            td_2_sa_h=None,
            td_2_sa_dropout=None,
            td_2_lstm_h=None,
            td_2_lstm_num_layers=None,
            td_2_lstm_dropout=None,
            td_2_lstm_bidirectional=None,

            pool='att',
            pool_att_h=128,
            pool_att_dropout=0.1,

            ):
        super().__init__()

        self.name = 'NISQA_DIM'

        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1,
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,
            )

        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
            )

        self.time_dependency_2 = TimeDependency(
            input_size=self.time_dependency.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
            )

        pool = Pooling(
            self.time_dependency.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
            )

        self.pool_layers = self._get_clones(pool, 5)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, x, n_wins):

        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        x, n_wins = self.time_dependency_2(x, n_wins)
        out = [mod(x, n_wins) for mod in self.pool_layers]
        out = torch.cat(out, dim=1)

        return out


#%% Framewise
class Framewise(nn.Module):
    '''
    Framewise: The main framewise module. It loads either a CNN or feed-forward
    network for framewise modelling of the Mel-spec segments. This module can
    also be skipped by loading the SkipCNN module. There are two CNN modules
    available. AdaptCNN with adaptive maxpooling and the StandardCNN module.
    However, they could also be replaced with new modules, such as PyTorch
    implementations of ResNet or Alexnet.
    '''
    def __init__(
        self,
        cnn_model,
        ms_seg_length=15,
        ms_n_mels=48,
        c_out_1=16,
        c_out_2=32,
        c_out_3=64,
        kernel_size=3,
        dropout=0.2,
        pool_1=[24,7],
        pool_2=[12,5],
        pool_3=[6,3],
        fc_out_h=None,
        ):
        super().__init__()

        if cnn_model=='adapt':
            self.model = AdaptCNN(
                input_channels=1,
                c_out_1=c_out_1,
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size,
                dropout=dropout,
                pool_1=pool_1,
                pool_2=pool_2,
                pool_3=pool_3,
                fc_out_h=fc_out_h,
                )
        elif cnn_model=='standard':
            assert ms_n_mels == 48, "ms_n_mels is {} and should be 48, use adaptive model or change ms_n_mels".format(ms_n_mels)
            assert ms_seg_length == 15, "ms_seg_len is {} should be 15, use adaptive model or change ms_seg_len".format(ms_seg_length)
            assert ((kernel_size == 3) or (kernel_size == (3,3))), "cnn_kernel_size is {} should be 3, use adaptive model or change cnn_kernel_size".format(kernel_size)
            self.model = StandardCNN(
                input_channels=1,
                c_out_1=c_out_1,
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size,
                dropout=dropout,
                fc_out_h=fc_out_h,
                )
        elif cnn_model=='dff':
            self.model = DFF(ms_seg_length, ms_n_mels, dropout, fc_out_h)
        elif (cnn_model is None) or (cnn_model=='skip'):
            self.model = SkipCNN(ms_seg_length, ms_n_mels, fc_out_h)
        else:
            raise NotImplementedError('Framwise model not available')

    def forward(self, x, n_wins):
        (bs, length, channels, height, width) = x.shape
        x_packed = pack_padded_sequence(
                x,
                n_wins.cpu(),
                batch_first=True,
                enforce_sorted=False
                )
        x = self.model(x_packed.data)
        x = x_packed._replace(data=x)
        x, _ = pad_packed_sequence(
            x,
            batch_first=True,
            padding_value=0.0,
            total_length=n_wins.max())
        return x

class SkipCNN(nn.Module):
    '''
    SkipCNN: Can be used to skip the framewise modelling stage and directly
    apply an LSTM or Self-Attention network.
    '''
    def __init__(
        self,
        cnn_seg_length,
        ms_n_mels,
        fc_out_h
        ):
        super().__init__()

        self.name = 'SkipCNN'
        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length*ms_n_mels
        self.bn = nn.BatchNorm2d( 1 )

        if fc_out_h is not None:
            self.linear = nn.Linear(self.fan_in, fc_out_h)
            self.fan_out = fc_out_h
        else:
            self.linear = nn.Identity()
            self.fan_out = self.fan_in

    def forward(self, x):
        x = self.bn(x)
        x = x.view(-1, self.fan_in)
        x = self.linear(x)
        return x

class DFF(nn.Module):
    '''
    DFF: Deep Feed-Forward network that was used as baseline framwise model as
    comparision to the CNN.
    '''
    def __init__(self,
                 cnn_seg_length,
                 ms_n_mels,
                 dropout,
                 fc_out_h=4096,
                 ):
        super().__init__()
        self.name = 'DFF'

        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h
        self.fan_out = fc_out_h

        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length*ms_n_mels

        self.lin1 = nn.Linear(self.fan_in, self.fc_out_h)
        self.lin2 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin3 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin4 = nn.Linear(self.fc_out_h, self.fc_out_h)

        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d( self.fc_out_h )
        self.bn3 = nn.BatchNorm1d( self.fc_out_h )
        self.bn4 = nn.BatchNorm1d( self.fc_out_h )
        self.bn5 = nn.BatchNorm1d( self.fc_out_h )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.bn1(x)
        x = x.view(-1, self.fan_in)

        x = F.relu( self.bn2( self.lin1(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn3( self.lin2(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.lin3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn5( self.lin4(x) ) )

        return x


class AdaptCNN(nn.Module):
    '''
    AdaptCNN: CNN with adaptive maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module is more flexible
    than the StandardCNN that requires a fixed input dimension of 48x15.
    '''
    def __init__(self,
                 input_channels,
                 c_out_1,
                 c_out_2,
                 c_out_3,
                 kernel_size,
                 dropout,
                 pool_1,
                 pool_2,
                 pool_3,
                 fc_out_h=20,
                 ):
        super().__init__()
        self.name = 'CNN_adapt'

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.pool_3 = pool_3
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        # Set kernel width of last conv layer to last pool width to
        # downsample width to one.
        self.kernel_size_last = (self.kernel_size[0], self.pool_3[1])

        # kernel_size[1]=1 can be used for seg_length=1 -> corresponds to
        # 1D conv layer, no width padding needed.
        if self.kernel_size[1] == 1:
            self.cnn_pad = (1,0)
        else:
            self.cnn_pad = (1,1)

        self.conv1 = nn.Conv2d(
                self.input_channels,
                self.c_out_1,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn1 = nn.BatchNorm2d( self.conv1.out_channels )

        self.conv2 = nn.Conv2d(
                self.conv1.out_channels,
                self.c_out_2,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn2 = nn.BatchNorm2d( self.conv2.out_channels )

        self.conv3 = nn.Conv2d(
                self.conv2.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn3 = nn.BatchNorm2d( self.conv3.out_channels )

        self.conv4 = nn.Conv2d(
                self.conv3.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn4 = nn.BatchNorm2d( self.conv4.out_channels )

        self.conv5 = nn.Conv2d(
                self.conv4.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn5 = nn.BatchNorm2d( self.conv5.out_channels )

        self.conv6 = nn.Conv2d(
                self.conv5.out_channels,
                self.c_out_3,
                self.kernel_size_last,
                padding = (1,0))

        self.bn6 = nn.BatchNorm2d( self.conv6.out_channels )

        if self.fc_out_h:
            self.fc = nn.Linear(self.conv6.out_channels * self.pool_3[0], self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.pool_3[0])

    def forward(self, x):

        x = F.relu( self.bn1( self.conv1(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_1))

        x = F.relu( self.bn2( self.conv2(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_2))

        x = self.dropout(x)
        x = F.relu( self.bn3( self.conv3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.conv4(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_3))

        x = self.dropout(x)
        x = F.relu( self.bn5( self.conv5(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn6( self.conv6(x) ) )
        x = x.view(-1, self.conv6.out_channels * self.pool_3[0])

        if self.fc_out_h:
            x = self.fc( x )
        return x

class StandardCNN(nn.Module):
    '''
    StandardCNN: CNN with fixed maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module requires a fixed
    input dimension of 48x15.
    '''
    def __init__(
        self,
        input_channels,
        c_out_1,
        c_out_2,
        c_out_3,
        kernel_size,
        dropout,
        fc_out_h=None
        ):
        super().__init__()

        self.name = 'CNN_standard'

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_size = 2
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.output_width = 2 # input width 15 pooled 3 times
        self.output_height = 6 # input height 48 pooled 3 times

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        self.pool_first = nn.MaxPool2d(
                self.pool_size,
                stride = self.pool_size,
                padding = (0,1))

        self.pool = nn.MaxPool2d(
                self.pool_size,
                stride = self.pool_size,
                padding = 0)

        self.conv1 = nn.Conv2d(
                self.input_channels,
                self.c_out_1,
                self.kernel_size,
                padding = 1)

        self.bn1 = nn.BatchNorm2d( self.conv1.out_channels )

        self.conv2 = nn.Conv2d(
                self.conv1.out_channels,
                self.c_out_2,
                self.kernel_size,
                padding = 1)

        self.bn2 = nn.BatchNorm2d( self.conv2.out_channels )


        self.conv3 = nn.Conv2d(
                self.conv2.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn3 = nn.BatchNorm2d( self.conv3.out_channels )

        self.conv4 = nn.Conv2d(
                self.conv3.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn4 = nn.BatchNorm2d( self.conv4.out_channels )

        self.conv5 = nn.Conv2d(
                self.conv4.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn5 = nn.BatchNorm2d( self.conv5.out_channels )

        self.conv6 = nn.Conv2d(
                self.conv5.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn6 = nn.BatchNorm2d( self.conv6.out_channels )

        if self.fc_out_h:
            self.fc_out = nn.Linear(self.conv6.out_channels * self.output_height * self.output_width, self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.output_height * self.output_width)

    def forward(self, x):

        x = F.relu( self.bn1( self.conv1(x) ) )
        x = self.pool_first( x )

        x = F.relu( self.bn2( self.conv2(x) ) )
        x = self.pool( x )

        x = self.dropout(x)
        x = F.relu( self.bn3( self.conv3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.conv4(x) ) )
        x = self.pool( x )

        x = self.dropout(x)
        x = F.relu( self.bn5( self.conv5(x) ) )
        x = self.dropout(x)

        x = F.relu( self.bn6( self.conv6(x) ) )

        x = x.view(-1, self.conv6.out_channels * self.output_height * self.output_width)

        if self.fc_out_h:
            x = self.fc_out( x )

        return x

#%% Time Dependency
class TimeDependency(nn.Module):
    '''
    TimeDependency: The main time-dependency module. It loads either an LSTM
    or self-attention network for time-dependency modelling of the framewise
    features. This module can also be skipped.
    '''
    def __init__(self,
                 input_size,
                 td='self_att',
                 sa_d_model=512,
                 sa_nhead=8,
                 sa_pos_enc=None,
                 sa_num_layers=6,
                 sa_h=2048,
                 sa_dropout=0.1,
                 lstm_h=128,
                 lstm_num_layers=1,
                 lstm_dropout=0,
                 lstm_bidirectional=True,
                 ):
        super().__init__()

        if td=='self_att':
            self.model = SelfAttention(
                input_size=input_size,
                d_model=sa_d_model,
                nhead=sa_nhead,
                pos_enc=sa_pos_enc,
                num_layers=sa_num_layers,
                sa_h=sa_h,
                dropout=sa_dropout,
                activation="relu"
                )
            self.fan_out = sa_d_model

        elif td=='lstm':
            self.model = LSTM(
                 input_size,
                 lstm_h=lstm_h,
                 num_layers=lstm_num_layers,
                 dropout=lstm_dropout,
                 bidirectional=lstm_bidirectional,
                 )
            self.fan_out = self.model.fan_out

        elif (td is None) or (td=='skip'):
            self.model = self._skip
            self.fan_out = input_size
        else:
            raise NotImplementedError('Time dependency option not available')

    def _skip(self, x, n_wins):
        return x, n_wins

    def forward(self, x, n_wins):
        x, n_wins = self.model(x, n_wins)
        return x, n_wins

class LSTM(nn.Module):
    '''
    LSTM: The main LSTM module that can be used as a time-dependency model.
    '''
    def __init__(self,
                 input_size,
                 lstm_h=128,
                 num_layers=1,
                 dropout=0.1,
                 bidirectional=True
                 ):
        super().__init__()

        self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = lstm_h,
                num_layers = num_layers,
                dropout = dropout,
                batch_first = True,
                bidirectional = bidirectional
                )

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        self.fan_out = num_directions*lstm_h

    def forward(self, x, n_wins):

        x = pack_padded_sequence(
                x,
                n_wins.cpu(),
                batch_first=True,
                enforce_sorted=False
                )

        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]

        x, _ = pad_packed_sequence(
            x,
            batch_first=True,
            padding_value=0.0,
            total_length=n_wins.max())

        return x, n_wins

class SelfAttention(nn.Module):
    '''
    SelfAttention: The main SelfAttention module that can be used as a
    time-dependency model.
    '''
    def __init__(self,
                 input_size,
                 d_model=512,
                 nhead=8,
                 pool_size=3,
                 pos_enc=None,
                 num_layers=6,
                 sa_h=2048,
                 dropout=0.1,
                 activation="relu"
                 ):
        super().__init__()

        encoder_layer = SelfAttentionLayer(d_model, nhead, pool_size, sa_h, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear = nn.Linear(input_size, d_model)

        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        if pos_enc:
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoder = nn.Identity()

        self._reset_parameters()

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, n_wins=None):
        src = self.linear(src)
        output = src.transpose(1,0)
        output = self.norm1(output)
        output = self.pos_encoder(output)

        for mod in self.layers:
            output, n_wins = mod(output, n_wins=n_wins)
        return output.transpose(1,0), n_wins

class SelfAttentionLayer(nn.Module):
    '''
    SelfAttentionLayer: The SelfAttentionLayer that is used by the
    SelfAttention module.
    '''
    def __init__(self, d_model, nhead, pool_size=1, sa_h=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, sa_h)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(sa_h, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

    def forward(self, src, n_wins=None):

        if n_wins is not None:
            mask = ~((torch.arange(src.shape[0])[None, :]).to(src.device) < n_wins[:, None].to(torch.long).to(src.device))
        else:
            mask = None

        src2 = self.self_attn(src, src, src, key_padding_mask=mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src, n_wins

class PositionalEncoding(nn.Module):
    '''
    PositionalEncoding: PositionalEncoding taken from the PyTorch Transformer
    tutorial. Can be applied to the SelfAttention module. However, it did not
    improve the results in previous experiments.
    '''
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#%% Pooling
class Pooling(nn.Module):
    '''
    Pooling: Main Pooling module. It can load either attention-pooling, average
    pooling, maxpooling, or last-step pooling. In case of bidirectional LSTMs
    last-step-bi pooling should be used instead of last-step pooling.
    '''
    def __init__(self,
                 d_input,
                 output_size=1,
                 pool='att',
                 att_h=None,
                 att_dropout=0,
                 ):
        super().__init__()

        if pool=='att':
            if att_h is None:
                self.model = PoolAtt(d_input, output_size)
            else:
                self.model = PoolAttFF(d_input, output_size, h=att_h, dropout=att_dropout)
        elif pool=='last_step_bi':
            self.model = PoolLastStepBi(d_input, output_size)
        elif pool=='last_step':
            self.model = PoolLastStep(d_input, output_size)
        elif pool=='max':
            self.model = PoolMax(d_input, output_size)
        elif pool=='avg':
            self.model = PoolAvg(d_input, output_size)
        else:
            raise NotImplementedError('Pool option not available')

    def forward(self, x, n_wins):
        return self.model(x, n_wins)

class PoolLastStepBi(nn.Module):
    '''
    PoolLastStepBi: last step pooling for the case of bidirectional LSTM
    '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, n_wins=None):
        x = x.view(x.shape[0], n_wins.max(), 2, x.shape[-1]//2)
        x = torch.cat(
            (x[torch.arange(x.shape[0]), n_wins.type(torch.long)-1, 0, :],
            x[:,0,1,:]),
            dim=1
            )
        x = self.linear(x)
        return x

class PoolLastStep(nn.Module):
    '''
    PoolLastStep: last step pooling can be applied to any one-directional
    sequence.
    '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, n_wins=None):
        x = x[torch.arange(x.shape[0]), n_wins.type(torch.long)-1]
        x = self.linear(x)
        return x

class PoolAtt(torch.nn.Module):
    '''
    PoolAtt: Attention-Pooling module.
    '''
    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear1 = nn.Linear(d_input, 1)
        self.linear2 = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):

        att = self.linear1(x)

        att = att.transpose(2,1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)

        x = self.linear2(x)

        return x

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''
    def __init__(self, d_input, output_size, h, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_input, h)
        self.linear2 = nn.Linear(h, 1)

        self.linear3 = nn.Linear(d_input, output_size)

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_wins):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)

        x = self.linear3(x)

        return x

class PoolAvg(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''
    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):

        mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, 0)

        x = torch.div(x.sum(1), n_wins.unsqueeze(1))

        x = self.linear(x)

        return x

class PoolMax(torch.nn.Module):
    '''
    PoolMax: Max-pooling that consideres masked time-steps.
    '''
    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):

        mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, float("-Inf"))

        x = x.max(1)[0]

        x = self.linear(x)

        return x


#%% Spectrograms
def segment_specs(file_path, x, seg_length, seg_hop=1, max_length=None):
    '''
    Segment a spectrogram into "seg_length" wide spectrogram segments.
    Instead of using only the frequency bin of the current time step,
    the neighboring bins are included as input to the CNN. For example
    for a seg_length of 7, the previous 3 and the follwing 3 frequency
    bins are included.

    A spectrogram with input size [H x W] will be segmented to:
    [W-(seg_length-1) x C x H x seg_length], where W is the width of the
    original mel-spec (corresponding to the length of the speech signal),
    H is the height of the mel-spec (corresponding to the number of mel bands),
    C is the number of CNN input Channels (always one in our case).
    '''
    if seg_length % 2 == 0:
        raise ValueError('seg_length must be odd! (seg_lenth={})'.format(seg_length))
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    n_wins = x.shape[1]-(seg_length-1)
    if n_wins < 1:
        raise ValueError(
            f"Sample too short. Only {x.shape[1]} windows available but seg_length={seg_length}. "
            f"Consider zero padding the audio sample. File: {file_path}"
        )

    # broadcast magic to segment melspec
    idx1 = torch.arange(seg_length)
    idx2 = torch.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose(1,0)[idx3,:].unsqueeze(1).transpose(3,2)

    if seg_hop>1:
        x = x[::seg_hop,:]
        n_wins = int(np.ceil(n_wins/seg_hop))

    if max_length is not None:
        if max_length < n_wins:
            raise ValueError('n_wins {} > max_length {} --- {}. Increase max window length ms_max_segments!'.format(n_wins, max_length, file_path))
        x_padded = torch.zeros((max_length, x.shape[1], x.shape[2], x.shape[3]))
        x_padded[:n_wins,:] = x
        x = x_padded

    return x, np.array(n_wins)

def get_librosa_melspec(
    file_path,
    sr=48e3,
    n_fft=1024,
    hop_length=80,
    win_length=170,
    n_mels=32,
    fmax=16e3,
    ms_channel=None,
    ):
    '''
    Calculate mel-spectrograms with Librosa.
    '''
    # Calc spec
    try:
        if ms_channel is not None:
            y, sr = lb.load(file_path, sr=sr, mono=False)
            if len(y.shape)>1:
                y = y[ms_channel, :]
        else:
            y, sr = lb.load(file_path, sr=sr)
    except:
        raise ValueError('Could not load file {}'.format(file_path))

    hop_length = int(sr * hop_length)
    win_length = int(sr * win_length)

    S = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=1.0,

        n_mels=n_mels,
        fmin=0.0,
        fmax=fmax,
        htk=False,
        norm='slaney',
        )

    spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
    return spec

def waveform_to_melspec(
    waveform,
    sr=48000,
    n_fft=1024,
    hop_length=80,
    win_length=170,
    n_mels=32,
    fmax=16000,
    ):
    '''
    Convert waveform (numpy array or torch tensor) to mel-spectrogram.
    '''
    # Convert to numpy if needed
    if torch.is_tensor(waveform):
        y = waveform.cpu().numpy()
    else:
        y = waveform

    # Ensure 1D
    if len(y.shape) > 1:
        y = y.flatten()

    # hop_length and win_length are in seconds
    hop_length = int(sr * hop_length)
    win_length = int(sr * win_length)

    S = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=1.0,
        n_mels=n_mels,
        fmin=0.0,
        fmax=fmax,
        htk=False,
        norm='slaney',
    )

    spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
    return spec


#%% NisqaModel - High-level inference interface
class NisqaModel:
    """
    NISQA Model for speech quality prediction (inference only).

    Usage:
        model = NisqaModel(model_path='nisqa/weights/nisqa.tar')
        result = model(waveform=audio_array)  # or filepath='path/to/audio.wav'
    """

    def __init__(self, model_path='nisqa/weights/nisqa.tar', device=None):
        """
        Args:
            model_path: Path to pretrained model (.tar file)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load checkpoint
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)

        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model arguments
        self.args = checkpoint['args']

        # Build model_args dict for NISQA_DIM
        model_args = {
            'ms_seg_length': self.args['ms_seg_length'],
            'ms_n_mels': self.args['ms_n_mels'],

            'cnn_model': self.args['cnn_model'],
            'cnn_c_out_1': self.args['cnn_c_out_1'],
            'cnn_c_out_2': self.args['cnn_c_out_2'],
            'cnn_c_out_3': self.args['cnn_c_out_3'],
            'cnn_kernel_size': self.args['cnn_kernel_size'],
            'cnn_dropout': self.args['cnn_dropout'],
            'cnn_pool_1': self.args['cnn_pool_1'],
            'cnn_pool_2': self.args['cnn_pool_2'],
            'cnn_pool_3': self.args['cnn_pool_3'],
            'cnn_fc_out_h': self.args['cnn_fc_out_h'],

            'td': self.args['td'],
            'td_sa_d_model': self.args['td_sa_d_model'],
            'td_sa_nhead': self.args['td_sa_nhead'],
            'td_sa_pos_enc': self.args['td_sa_pos_enc'],
            'td_sa_num_layers': self.args['td_sa_num_layers'],
            'td_sa_h': self.args['td_sa_h'],
            'td_sa_dropout': self.args['td_sa_dropout'],
            'td_lstm_h': self.args['td_lstm_h'],
            'td_lstm_num_layers': self.args['td_lstm_num_layers'],
            'td_lstm_dropout': self.args['td_lstm_dropout'],
            'td_lstm_bidirectional': self.args['td_lstm_bidirectional'],

            'td_2': self.args['td_2'],
            'td_2_sa_d_model': self.args['td_2_sa_d_model'],
            'td_2_sa_nhead': self.args['td_2_sa_nhead'],
            'td_2_sa_pos_enc': self.args['td_2_sa_pos_enc'],
            'td_2_sa_num_layers': self.args['td_2_sa_num_layers'],
            'td_2_sa_h': self.args['td_2_sa_h'],
            'td_2_sa_dropout': self.args['td_2_sa_dropout'],
            'td_2_lstm_h': self.args['td_2_lstm_h'],
            'td_2_lstm_num_layers': self.args['td_2_lstm_num_layers'],
            'td_2_lstm_dropout': self.args['td_2_lstm_dropout'],
            'td_2_lstm_bidirectional': self.args['td_2_lstm_bidirectional'],

            'pool': self.args['pool'],
            'pool_att_h': self.args['pool_att_h'],
            'pool_att_dropout': self.args['pool_att_dropout'],
        }

        # Initialize model
        self.model = NISQA_DIM(**model_args)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, waveforms, sr=None):
        """
        Predict speech quality for a batch of waveforms using a single GPU forward pass.

        Args:
            waveforms: list of np.ndarray audio arrays (float32, at 48kHz)
            sr: sample rate (default: 48000 or model's ms_sr)

        Returns:
            list of dicts with keys: 'mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred'
        """
        from concurrent.futures import ThreadPoolExecutor

        if not waveforms:
            return []

        sample_rate = sr
        if sample_rate is None:
            sample_rate = self.args.get('ms_sr') or 48000

        seg_length = self.args['ms_seg_length']
        seg_hop = self.args.get('ms_seg_hop_length', 1)

        # Step 1: compute mel-spectrograms in parallel (CPU-bound)
        def _spec(w):
            return waveform_to_melspec(
                w.astype(np.float32),
                sr=sample_rate,
                n_fft=self.args['ms_n_fft'],
                hop_length=self.args['ms_hop_length'],
                win_length=self.args['ms_win_length'],
                n_mels=self.args['ms_n_mels'],
                fmax=self.args['ms_fmax'],
            )

        with ThreadPoolExecutor() as pool:
            specs = list(pool.map(_spec, waveforms))

        # Step 2: segment + collect n_wins
        segs = []
        n_wins_list = []
        valid_mask = []
        for spec in specs:
            try:
                seg, n_w = segment_specs('waveform', spec, seg_length, seg_hop=seg_hop)
                segs.append(seg)
                n_wins_list.append(int(n_w))
                valid_mask.append(True)
            except Exception:
                # Too-short audio: append placeholder
                segs.append(None)
                n_wins_list.append(0)
                valid_mask.append(False)

        # Step 3: pad to max length and stack
        max_wins = max((n for n, v in zip(n_wins_list, valid_mask) if v), default=1)

        padded = []
        for seg, n_w, valid in zip(segs, n_wins_list, valid_mask):
            if not valid:
                # Create zero-filled placeholder with correct shape
                dummy_shape = segs[next(i for i, v in enumerate(valid_mask) if v)].shape[1:]
                padded.append(torch.zeros(max_wins, *dummy_shape))
                continue
            if n_w < max_wins:
                pad = torch.zeros(max_wins - n_w, *seg.shape[1:])
                seg = torch.cat([seg, pad], dim=0)
            padded.append(seg)

        batch = torch.stack(padded, dim=0)  # [B, max_wins, 1, n_mels, seg_length]
        n_wins_tensor = torch.tensor(
            [max(n, 1) for n in n_wins_list], dtype=torch.long
        )

        batch = batch.to(self.device)
        n_wins_tensor = n_wins_tensor.to(self.device)

        # Step 4: single batch forward pass
        with torch.no_grad():
            predictions = self.model(batch, n_wins_tensor)

        predictions = predictions.cpu().numpy()

        results = []
        for i in range(len(waveforms)):
            if valid_mask[i]:
                results.append({
                    'mos_pred': float(predictions[i][0]),
                    'noi_pred': float(predictions[i][1]),
                    'dis_pred': float(predictions[i][2]),
                    'col_pred': float(predictions[i][3]),
                    'loud_pred': float(predictions[i][4]),
                })
            else:
                results.append({
                    'mos_pred': 0.0, 'noi_pred': 0.0,
                    'dis_pred': 0.0, 'col_pred': 0.0, 'loud_pred': 0.0,
                })
        return results

    def __call__(self, waveform=None, filepath=None, sr=None):
        """
        Predict speech quality.

        Args:
            waveform: np.ndarray or torch.Tensor audio (optional)
            filepath: str path to audio file (optional)
            sr: sample rate of waveform in Hz (optional, defaults to 48000 if not specified)

        Returns:
            dict with keys: 'mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred'
        """
        if filepath is None and waveform is None:
            raise ValueError("Either filepath or waveform must be provided")

        if filepath is not None and waveform is not None:
            raise ValueError("Only one of filepath or waveform should be provided")

        # Get mel-spectrogram
        if filepath is not None:
            spec = get_librosa_melspec(
                filepath,
                sr=self.args['ms_sr'],
                n_fft=self.args['ms_n_fft'],
                hop_length=self.args['ms_hop_length'],
                win_length=self.args['ms_win_length'],
                n_mels=self.args['ms_n_mels'],
                fmax=self.args['ms_fmax'],
                ms_channel=self.args.get('ms_channel', None)
            )
        else:
            # Use provided sr, or default to model's sr, or 48kHz if both are None
            if sr is not None:
                sample_rate = sr
            elif self.args['ms_sr'] is not None:
                sample_rate = self.args['ms_sr']
            else:
                sample_rate = 48000

            spec = waveform_to_melspec(
                waveform,
                sr=sample_rate,
                n_fft=self.args['ms_n_fft'],
                hop_length=self.args['ms_hop_length'],
                win_length=self.args['ms_win_length'],
                n_mels=self.args['ms_n_mels'],
                fmax=self.args['ms_fmax']
            )

        # Segment the spectrogram
        spec_seg, n_wins = segment_specs(
            filepath if filepath else 'waveform',
            spec,
            self.args['ms_seg_length'],
            seg_hop=self.args.get('ms_seg_hop_length', 1)
        )

        # Add batch dimension
        spec_seg = spec_seg.unsqueeze(0)
        # n_wins is a numpy scalar, convert to tensor with shape [1]
        n_wins = torch.tensor([int(n_wins)])

        # Move to device
        spec_seg = spec_seg.to(self.device)
        n_wins = n_wins.to(self.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model(spec_seg, n_wins)

        # Convert to numpy and extract predictions
        predictions = predictions.cpu().numpy()[0]

        return {
            'mos_pred': float(predictions[0]),
            'noi_pred': float(predictions[1]),
            'dis_pred': float(predictions[2]),
            'col_pred': float(predictions[3]),
            'loud_pred': float(predictions[4])
        }
