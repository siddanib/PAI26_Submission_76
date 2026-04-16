import torch
from model_reflect import Flow_Transformer

### Set data type
t_dtype = torch.float32
torch.set_default_dtype(t_dtype)

t_device = "cpu"
torch.set_default_device(t_device)

torch.autograd.set_grad_enabled(False)

flow = Flow_Transformer(input_N_dim=2, input_F_dim=1,
                        d_model = 256, nhead=4,
                        num_encoder_layers=2,
                        num_decoder_layers=2,
                        dropout = 0.,
                        max_len=50, d_embed=256,
                        n_layers=2,act_func=torch.nn.ReLU(),
                        residual_con=True)

model_folder = "./"

chpt_fl = torch.load(model_folder +
                   "flow_model.tar",
                   map_location=torch.device(t_device),
                   weights_only=True)
flow.load_state_dict(chpt_fl['model_state_dict'])
# Dropout needs to be turned off for symmetry
flow.train(False)
flow.to(t_dtype)
flow.to(t_device)

scripted_flow = torch.jit.script(flow)
scripted_flow =  torch.jit.optimize_for_inference(scripted_flow)

input_N_dim = 2
input_F_dim = 1
seq_len = 10
batch_sz = 10
x_N = torch.randn(batch_sz, seq_len,input_N_dim)
x_F = torch.randn(batch_sz, seq_len-1,input_F_dim)
x_t = torch.randn(batch_sz, 1, 1)
t = torch.randn(batch_sz, 1, 1)

y_1 = flow(x_t, x_N, x_F, t)
print(y_1)

y_2 = scripted_flow(x_t, x_N, x_F, t)
print(y_2)
print(y_2.device)
print(y_2.dtype)

assert torch.allclose(y_1,y_2,atol=1.0e-6)

if t_dtype == torch.float64:
    model_name = "_double_"
else:
    model_name = "_single_"

if t_device == "cpu":
    model_name = model_name + "cpu"
else:
    model_name = model_name + "gpu"

torch.jit.save(scripted_flow,
               model_folder+"scripted_model"+model_name+".pt")
