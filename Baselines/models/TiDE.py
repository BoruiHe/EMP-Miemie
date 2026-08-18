import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True): 
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)
        
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out

#TiDE
class Model(nn.Module):  
    """
    paper: https://arxiv.org/pdf/2304.08424.pdf 
    """
    def __init__(self, configs, bias=True, feature_encode_dim=2): 
        super(Model, self).__init__()
        self.seq_len= configs.seq_len  #L 
        self.pred_len= configs.pred_len  #H 
        self.hidden_dim= configs.d_model
        self.res_hidden= configs.d_model 
        self.encoder_num= configs.e_layers
        self.decoder_num= configs.d_layers
        self.feature_encode_dim= feature_encode_dim
        self.decode_dim= configs.c_out
        self.temporalDecoderHidden= configs.d_ff
        dropout= configs.dropout
        
        self.feature_dim=1

        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, bias)
        self.encoders = nn.Sequential(ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias),*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.encoder_num-1)))

        self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len, dropout, bias))
        self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)
            
        
    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        feature = self.feature_encoder(batch_y_mark)
        hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.pred_len, self.decode_dim)
        dec_out = self.temporalDecoder(torch.cat([feature[:,self.seq_len:], decoded], dim=-1)).squeeze(-1) + self.residual_proj(x_enc)
        
        # De-Normalization 
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        '''x_mark_enc is the exogenous dynamic feature described in the original paper'''
        if batch_y_mark is None:
            batch_y_mark = torch.zeros((x_enc.shape[0], self.seq_len+self.pred_len, self.feature_dim)).to(x_enc.device).detach()
        else:
            batch_y_mark = torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]],dim=1)
        dec_out = torch.stack([self.forecast(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark) for feature in range(x_enc.shape[-1])],dim=-1)
        return dec_out # [B, L, D]
    

if __name__ == "__main__":
    from torchinfo import summary
    import torch as tr
    import argparse


    parser = argparse.ArgumentParser(description='Good luck Miemie')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, help='model name, options: [Autoformer, DLinear]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='RP', help='dataset type')
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
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    # Autoformer 
    parser.add_argument('--enc_in', type=int, default=25, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=25, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=25, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args('--is_training 1 --model_id 1 --model 1 --data 1 --d_model 256 --d_ff 256 --dropout 0.3 --learning_rate 0.1'.split())
    
    model = Model(args)
    bs, seq_len, ps, dim = 4, 60, 30, 25

    x = tr.randn((bs, seq_len, dim))
    x_mark = tr.randn((bs,seq_len, 1))
    y_mark = tr.randn((bs,ps, 1))

    summary(model, input_size=[(bs, seq_len, dim), (bs,seq_len, 1), (bs,ps, 1), (bs,ps, 1)])
    out = model(x, x_mark, x, y_mark)
    print(model)