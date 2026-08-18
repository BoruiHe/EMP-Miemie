import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Flowformer_layers import Encoder, EncoderLayer
from layers.Flow_Attention import Flow_Attention, AttentionLayer


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learned":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.feat_dim = 25
        self.num_classes = configs.num_class
        self.freeze = False
        self.d_model = configs.d_model
        self.project_inp = nn.Linear(self.feat_dim, configs.d_model)
        self.pos_enc = get_pos_encoder(configs.embed)(configs.d_model, dropout=configs.dropout * (1.0 - self.freeze), max_len=configs.max_len)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Flow_Attention(attention_dropout=configs.dropout),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.output_layer = nn.Linear(configs.d_model, self.feat_dim)
        self.act = _get_activation_fn(configs.activation)
        self.dropout = nn.Dropout(configs.dropout)
        self.output_layer = self.build_output_module(configs.d_model, configs.max_len, self.num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, x_enc, padding_masks, dummy_A, dummy_B):
        inp = x_enc.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        inp = inp.permute(1, 0, 2)

        enc_out = self.encoder(inp)
        enc_out = enc_out.permute(1, 0, 2)
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout(output)

        # Output
        # output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output


if __name__ == "__main__":
    import argparse


    seeds = [608, 1247, 3224] 
    parser = argparse.ArgumentParser(description='Good luck Miemie')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, help='model description')
    parser.add_argument('--model', type=str, required=True, help='model name, options: [Autoformer, DLinear, TiDE, TimeMixer, TimeXer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M]; M:multivariate predict multivariate.')
    parser.add_argument('--fps', type=str, default=None,
                        help='freq for the VP, options: [16, 30]')
    parser.add_argument('--freq', type=str, default='ts',
                        help='freq for time features encoding. No other choices unless more datasets are included')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=30, help='start token length')
    parser.add_argument('--pred_len', type=int, default=30, help='prediction sequence length')
    parser.add_argument('--exogenous', action='store_true', help='whether the model takes exogenous variables, False for no, default is False.')
    parser.add_argument('--do_cdv', action='store_true', help='whether to do cross data validation, True for yes, default is False.')

    # Flowformer
    parser.add_argument('--max_len', type=int, default=60, help='maximum of seq_len')
    # TimeMixer
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--down_sampling_layers', type=int, default=1, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg', help='down sampling method, only support avg, max, conv')
    # TimeXer
    parser.add_argument('--patch_len', type=int, default=24, help='patch length for short term forecasting')
    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate/channel (individually).')
    # Autoformer 
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    # Common parameters
    parser.add_argument('--enc_in', type=int, default=25, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=25, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=25, help='output size')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args('--is_training 1 --model_id 1 --model 1 --data 1 --dropout 0.1 --embed learned'.split())

    model = Model(args)
    x_enc = torch.randn(1, 60, 25)
    out = model(x_enc, x_enc)
    print(out)