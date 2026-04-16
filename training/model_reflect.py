import math
import numpy as np
import torch
import torch.nn as nn
from no_history_model import Flow_DeepONet 

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (N, S, E)
        S = x.size(1)
        return x + self.pe[0, :S, :] # broadcast over batch

# Learnable Time Encoding
class TimeEncoding(nn.Module):
    def __init__(self, d_embed, d_model, n_layers=2, act_func=nn.ReLU()):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_layers = n_layers
        self.act_func = act_func

        self.module_list = torch.nn.ModuleList([torch.nn.Linear(1,self.d_embed,
                                                bias=True)])
        self.module_list.append(self.act_func)

        for i in range(1, self.n_layers):
            self.module_list.append(torch.nn.Linear(
                self.d_embed,self.d_embed,bias=True))
            self.module_list.append(self.act_func)

        self.module_list.append(torch.nn.Linear(self.d_embed,
                                                self.d_model,bias=True))
    def forward (self, x):
        for lyr in self.module_list:
            x = lyr(x)
        return x

# A base class
class Flow_Base(nn.Module):
    def __init__ (self):
        super().__init__()

    def step (self, x_t, c_N, c_F, t_start, t_end):
        h = t_end - t_start
        return x_t + h*self(x_t, c_N, c_F, t_start)

    def sample (self, x_0, c_N, c_F, n_steps=100):
        dt = 1.0/n_steps
        for i_t in range(n_steps):
            t_start = torch.ones_like(x_0)*i_t*dt
            t_end = t_start + dt
            x_0 = self.step(x_0, c_N, c_F, t_start, t_end)
        return x_0

# Transformer-based model
class Flow_Transformer(Flow_Base):
    def __init__(self, input_N_dim=5, input_F_dim=1, d_model=64,
                 nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                 forecast_len=1, dim_feedforward=None, dropout=0.1,
                 max_len=1000, d_embed=64, n_layers=2, act_func=nn.ReLU(),
                 residual_con=True):
        super().__init__()
        self.input_N_dim = input_N_dim
        self.input_F_dim = input_F_dim
        self.d_model = d_model
        self.forecast_len = forecast_len

        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        ########################## Transformer related #############################
        # Embeddings
        self.input_N_proj = nn.Linear(input_N_dim, d_model)
        self.input_F_proj = nn.Linear(input_F_dim, d_model)
        self.target_proj  = nn.Linear(1, d_model)  # forecast values map to d_model

        # Type Embedding to distinguish between N and F
        self.type_embed = nn.Embedding(2, d_model)
        type_N = torch.LongTensor([0])
        self.register_buffer("type_N", type_N)
        type_F = torch.LongTensor([1])
        self.register_buffer("type_F", type_F)

        # Positional encoding 
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        # Time Encoding
        self.time_enc = TimeEncoding(d_embed=d_embed, d_model=d_model,
                                     n_layers=n_layers, act_func=act_func)

        # Encoder
        self.encoder_layer_N = nn.TransformerEncoderLayer(d_model, nhead,
                                                        dim_feedforward,
                                                        dropout=dropout,
                                                        batch_first=True)

        self.encoder_N = nn.TransformerEncoder(self.encoder_layer_N,
                                             num_layers=num_encoder_layers)

        self.encoder_layer_F = nn.TransformerEncoderLayer(d_model, nhead,
                                                        dim_feedforward,
                                                        dropout=dropout,
                                                        batch_first=True)

        self.encoder_F = nn.TransformerEncoder(self.encoder_layer_F,
                                             num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead,
                                                        dim_feedforward,
                                                        dropout=dropout,
                                                        batch_first=True)

        self.decoder = nn.TransformerDecoder(self.decoder_layer,
                                             num_layers=num_decoder_layers)

        # Output head: map to 1 value per forecast step
        self.out = nn.Linear(d_model, 1)
        ########################## End of Transformer related ###############
        ##### DeepONet - only for no history case ##########################
        self.no_history_model = Flow_DeepONet(cond_size=input_N_dim, output_size=1,
                                              n_layers=n_layers,
                                              layer_width=d_model,act_func=act_func,
                                              residual_con = residual_con,
                                              time_embedding=d_embed)
        ####################################################################

    def forward(self, tgt, src_N, src_F, t):
        out_org = self.forward_single(tgt, src_N, src_F, t)
        # Reflected version
        out_refl = self.forward_single(-tgt, src_N.flip(dims=[-1,]),
                                       -src_F, t)

        return 0.5*(out_org-out_refl)

    def forward_single(self, tgt, src_N, src_F, t):
        """
        tgt: (B, L, 1)  (L=1)
        src_N: (B, S, input_N_dim)
        src_F: (B, S-1, input_F_dim)
        t : (B, 1, 1)
        Returns: (B, L)
        """
        if src_N.size(dim=-2) == 1:
            out = self.no_history_model(tgt, src_N, t)
        else:
            # Encoder
            src_N_emb = self.input_N_proj(src_N) # (B, S, d_model)
            # Adding position embedding
            src_N_emb = self.pos_enc(src_N_emb)
            # Encoder for N
            memory_N = self.encoder_N(src_N_emb)
            ## Add type embedding
            memory_N = memory_N + self.type_embed(self.type_N.to(memory_N.device))
            src_F_emb = self.input_F_proj(src_F) # (B, S-1, d_model)
            # Adding position embedding
            src_F_emb = self.pos_enc(src_F_emb)
            # Encoder for F
            memory_F = self.encoder_F(src_F_emb)
            ## Add type embedding
            memory_F = memory_F + self.type_embed(self.type_F.to(memory_F.device))
            ## Concatenate along sequence dimension
            memory = torch.cat([memory_N, memory_F], dim=-2)
            # Decoder
            tgt_emb = self.target_proj(tgt)      # (B, L, d_model)
            # Adding time embedding
            tgt_emb = tgt_emb + self.time_enc(t)
            dec_out = self.decoder(tgt_emb, memory) # (B, L, d_model)
            out = self.out(dec_out)                 # (B, L, 1)
        return out

# Example usage
if __name__ == "__main__":
    #mdl = PositionalEncoding(10, 16)
    #print(mdl.pe)
    #mdl = TimeEncoding(10, 10)
    #print(mdl.module_list)
    input_N_dim = 4
    input_F_dim = 1
    seq_len = 1
    batch_sz = 10
    x_N = torch.randn(batch_sz, seq_len,input_N_dim)
    x_F = torch.randn(batch_sz, seq_len-1,input_F_dim)
    x_t = torch.randn(batch_sz, 1, 1)
    t = torch.randn(batch_sz, 1, 1)

    mdl  = Flow_Transformer(input_N_dim=input_N_dim,
                            input_F_dim=input_F_dim)
