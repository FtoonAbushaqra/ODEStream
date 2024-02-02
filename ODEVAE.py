
import seaborn as sns
sns.color_palette("bright")
import torch.nn as nn
import torch
from NeuralODE import LinearODEF, NNODEF, to_np, NeuralODE, ODEAdjoint,ODEF, ode_solve
from data import get_data, gen_batch

use_cuda = torch.cuda.is_available()
class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        #print("input_dim, ", input_dim)
        #print("hidden_dim, ", hidden_dim)
        #print("latent_dim, ", latent_dim)

        self.rnn = nn.GRU(input_dim + 1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(self, x, t):
        #print(x.shape)
        #print("t in RNNEncoder", t.shape)
        # Concatenate time to input
        t = t.clone()
        #print("t in RNNEncoder", t.shape)
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.
        xt = torch.cat((x, t), dim=-1)
       # print(xt.shape)

        _, h0 = self.rnn(xt.flip((0,)))  # Reversed
        # Compute latent dimension
        z0 = self.hid2lat(h0[0])
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]
        return z0_mean, z0_log_var


class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        func = NNODEF(latent_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0, t):
        zs = self.ode(z0, t, return_whole_sequence=False)

        hs = self.l2h(zs)
        xs = self.h2o(hs)
        #print ("true")
        #print (zs.shape, hs.shape, xs.shape )

        return xs


class ODEVAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, input_dim ):
        super(ODEVAE, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = RNNEncoder(input_dim, hidden_dim, latent_dim)

        self.decoder = NeuralODEDecoder(output_dim, hidden_dim, latent_dim)

    def forward(self, x, t, y, MAP=False):
        #print("x = ", x.shape)
       #print(t.shape)

        z_mean, z_log_var = self.encoder(x, t)
        if MAP:
            z = z_mean
        else:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)
        x_p = self.decoder(z, t)
        #print(x_p.shape)
        return x_p, z, z_mean, z_log_var

    def generate_with_seed(self, seed_x, t):
        seed_t_len = seed_x.shape[0]
        # print ("t ", t)
        # print ("seed_t_len", seed_t_len)
        # print ("T seed_t_len", t[:seed_t_len])

        z_mean, z_log_var = self.encoder(seed_x, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        return x_p

def vali(batch_size, val_x, val_y, val_samp_ts,lag,vae, noise_std):
    Vlosses = []
    Vmsev = []
    #print(val_x.shape, val_samp_ts.shape, val_y.shape)
    Vtrain_iter = gen_batch(batch_size, val_x, val_y, val_samp_ts,lag)
    for vxr, vt, vyr in Vtrain_iter:

        vmax_len =24
        #(vxr.shape, vt.shape, vyr.shape)
        vx_p, vz, vz_mean, vz_log_var = vae(vxr, vt, vyr)
        vkl_loss = -0.5 * torch.sum(1 + vz_log_var - vz_mean ** 2 - torch.exp(vz_log_var), -1)

        vloss = 0.5 * ((vyr - vx_p) ** 2).sum(-1).sum(0) / noise_std ** 2 + vkl_loss
        vloss = torch.mean(vloss)
        vloss /= vmax_len
        vMSEV = torch.mean((vyr - vx_p) ** 2)
        Vlosses.append(vloss.item())
        Vmsev.append(vMSEV.item())
        return Vmsev, Vlosses
