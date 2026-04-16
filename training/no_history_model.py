import torch

class Flow_DeepONet (torch.nn.Module):
    def __init__ (self,cond_size, output_size,
                 n_layers, layer_width, act_func = torch.nn.ReLU(),
                 residual_con=True, time_embedding=128):
        super().__init__()
        self.input_size_1  = output_size
        self.input_size_2  = cond_size
        self.input_size_3  = 1
        self.output_size = output_size
        self.n_layers    = n_layers
        self.layer_width = layer_width
        self.act_func    = act_func
        self.time_embedding = time_embedding

        self.module_list_1 = torch.nn.ModuleList([
                            building_block(self.input_size_1,self.layer_width,
                                           1,act_func,residual_con)])

        self.module_list_2 = torch.nn.ModuleList([
                            building_block(self.input_size_2,self.layer_width,
                                           1,act_func,residual_con)])

        self.module_list_3 = torch.nn.ModuleList([
                                SinusoidalTimeEmbedding(self.time_embedding)])
        self.module_list_3.append(building_block(self.time_embedding,self.layer_width,
                                           1,act_func,residual_con))

        for i in range(1, self.n_layers-1):
            self.module_list_1.append(
                            building_block(self.layer_width,self.layer_width,
                                           1,act_func,residual_con))
            self.module_list_2.append(
                            building_block(self.layer_width,self.layer_width,
                                           1,act_func,residual_con))
            self.module_list_3.append(
                            building_block(self.layer_width,self.layer_width,
                                           1,act_func,residual_con))

        self.combining_layer = building_block(self.layer_width,self.output_size,
                                           1,act_func,residual_con)

    def forward (self, x_t, c, t):
        for lyr in self.module_list_1:
            x_t = lyr(x_t)

        for lyr in self.module_list_2:
            c = lyr(c)

        for lyr in self.module_list_3:
            t = lyr(t)
        x = self.combining_layer(x_t*c*t)
        return x

class building_block (torch.nn.Module):
    def __init__ (self, input_size, output_size, n_layers=1,
                  act_func = torch.nn.ReLU(), residual_con=True):
        super().__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.act_func = act_func
        self.residual_con = residual_con

        self.module_list = torch.nn.ModuleList([
                            torch.nn.Linear(self.input_size,self.output_size,
                                            bias=True)])

        for i in range(1, self.n_layers):
            self.module_list.append(torch.nn.Linear(
                self.output_size,self.output_size,bias=True))

        self.module_list.append(torch.nn.Linear(self.output_size,
                                                self.output_size,bias=True))
        self.residual_layer = None
        if self.residual_con:
            self.residual_layer = torch.nn.Linear(self.input_size, self.output_size,
                                               bias=True)
        else:
            self.residual_layer = torch.nn.Identity()

    def forward (self, x):
        x1 = torch.clone(x)
        last = len(self.module_list) - 1
        for i, layer in enumerate(self.module_list):
            x1 = layer(x1)
            if i != last:
                x1 = self.act_func(x1)
        x1 = x1 + self.residual_layer(x)
        return x1

class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        """
        Args:
            t: Tensor of shape (..., 1)
        Returns:
            embedding: shape (..., embedding_dim)
        """
        half_dim = self.embedding_dim // 2
        # Compute the frequencies
        device = t.device
        exponents = torch.arange(half_dim, device=device) / half_dim
        ## Following https://github.com/cambridge-mlg/pdediff/blob/master/pdediff/nn/embedding.py#L9
        frequencies = 10000 ** exponents  # shape (half_dim,)
        # Compute embeddings
        args = t * frequencies
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

if __name__ == "__main__":
    #mdl = Flow_DeepONet(2,1,2,64, act_func=torch.nn.ELU())
    #x = torch.tensor([[0.5, 0.83]])
    #t = torch.tensor([[0.1]])
    #print(mdl(t, x, t))
    #print(mdl.step(t, x, t,t+0.005))
    #print(mdl.sample(t, x))
    embedding_layer = SinusoidalTimeEmbedding(embedding_dim=64)
    time_tensor = torch.tensor([[0.0], [0.5], [1.0]])  # Shape: (3, 1)
    embeddings = embedding_layer(time_tensor)

    print(embeddings.shape)  # Expected output: (3)
    print(embeddings)
