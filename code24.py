import seaborn as sns
sns.color_palette("bright")
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset
import torch
from torch import Tensor
import numpy as np
from ODEVAE import RNNEncoder, NeuralODEDecoder, ODEVAE, vali
from data import get_data, gen_batch
from TIL import LSTMModel, ConcatenationLayer ,CombinedModel
from streaming import StreamingDataLoader
import torch.optim as optim
import os
import psutil
from memory_profiler import profile
import time
use_cuda = torch.cuda.is_available()


lag = 24
datasetnane = 'ETTm1' #ETTm1 , ETTh1, ETTh2, WTH
task = 'm' #'ms' , 'm', 's'
flag = 'stream' # initial, stream

hidden_dim = 64
latent_dim = 64
noise_std = 0.002
savedmodelpath = "models/"
resultpath = "results/"
mempath = "memory/"
num_layers = 2
input_size =7
lr = 0.0001
lossfun = 4
reguler = 't' # t rguler or f irreguler
# loss 1 : 0.8 * ((yr - x_p) ** 2).sum(-1).sum(0) / noise_std ** 2 + kl_loss + l1
#loss 2 : 0.8 * ((yr - x_p) ** 2).sum(-1).sum(0) / noise_std ** 2 + kl_loss
#loss 3 :mse
if datasetnane == 'ECL':
    if task == 'm':
        output_size = 321
        input_size = 321
    if  task == 's':
        output_size = 1
        input_size = 1
if datasetnane in ['ETTm1', 'ETTh1', 'ETTh2' ]:
    if task == 'm':
        output_size = 7
        input_size = 7
    if task == 'ms':
        output_size = 1
        input_size = 7
    if task == 's':
        output_size = 1
        input_size = 1

if datasetnane == 'WTH':
    if task == 'm':
        output_size = 12
        input_size = 12
    if task == 'ms':
        output_size = 1
        input_size = 12
    if task == 's':
        output_size = 1
        input_size = 1



if flag=='initial':
    n_epochs = 1
    preload = False
    batch_size = 64
    patience = 10
else:
    n_epochs = 1
    batch_size = 1
    loadfile = "models\ETTm1_m_initial_64_64_2.pth"
    preload = True

vae = ODEVAE(output_size, hidden_dim, latent_dim, input_size)
if use_cuda:
    vae = vae.cuda()
optim = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=0.001)


if flag=='initial':
    x, y, val_x, val_y, samp_ts, val_samp_ts = get_data(datasetnane, flag, lag, task, reguler)
    best_val_loss = float('inf')
    for epoch_idx in range(n_epochs):
        losses1 = []
        losses2 = []
        losses3 = []
        train_iter = gen_batch(batch_size, x, y, samp_ts, lag)
        for xr, t, yr in train_iter:
            optim.zero_grad()
            if use_cuda:
                xr, t = xr.cuda(), t.cuda()
            #  print('t = ', t.shape)
            max_len = lag
            # print("real xr = ", xr.shape)
            # print("t in loop = ", xr.shape)
            # print("y in loop = ", yr.shape)

            x_p, z, z_mean, z_log_var = vae(xr, t, yr)
            #nt("x_p = ", x_p.shape)

            # yr = yr.reshape([8,1])
            # yr = torch.from_numpy(yr[:, :, ]).to(torch.float32)
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
            l1 = 0.01 * torch.abs(z_mean).sum(-1)
            lossL1 = 0.5 * ((yr - x_p) ** 2).sum(-1).sum(0) / noise_std ** 2 + kl_loss + l1
            loss = 0.5 * ((yr - x_p) ** 2).sum(-1).sum(0) / noise_std ** 2 + kl_loss

            lossL1= torch.mean(lossL1)
            lossL1 /= max_len

            loss = torch.mean(loss)
            loss /= max_len

            MSEV = torch.mean((yr - x_p) ** 2)

            if lossfun==1:
                lossL1.backward()
            if lossfun == 2:
                 loss.backward()
            if lossfun==3:
                MSEV.backward()
            optim.step()

            losses1.append(lossL1.item())
            losses2.append(loss.item())
            losses3.append(MSEV.item())

        vali_loss, vmse = vali(batch_size, val_x, val_y, val_samp_ts,lag,vae, noise_std)
        mean_val_loss = np.mean(vali_loss)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch_idx}")
            break

        print(f"Epoch {epoch_idx}")

        print("train: ",np.mean(losses1), np.mean(losses2), np.mean(losses3))
        print("val: ", mean_val_loss,  np.mean(vmse))

    print(np.mean(losses1), np.mean(losses2), np.mean(losses3))
    print(np.mean(vali_loss), np.mean(vmse))

    outputn= f"{datasetnane}_{task}_{flag}_{hidden_dim}_{latent_dim}_{n_epochs}_{lossfun}_{reguler}"
    torch.save(vae.state_dict(), savedmodelpath + outputn +".pth")

def calculate_memory_usage():
    # Measure memory usage
    memory_usage = psutil.Process().memory_info().rss
    current_memory = psutil.virtual_memory().used

    print(f"Memory Usage: {memory_usage} bytes")
    print(f"Current Memory Usage: {current_memory} bytes")
    print(f"Current Memory Usage: {current_memory / (1024 ** 3):.2f} GB")

     # Format the memory usage information
    memory_info_str = f"Memory Usage: {memory_usage} bytes\n"
    memory_info_str += f"Current Memory Usage: {current_memory} bytes\n"
    memory_info_str += f"Current Memory Usage: {current_memory / (1024 ** 3):.2f} GB\n"
    filename = f"{datasetnane}__memory_usage.txt"
    with open(mempath+filename, "a") as file:
        file.write(memory_info_str)



if flag=='stream':
    x, y,  samp_ts = get_data(datasetnane, flag, lag, task,reguler)
    lstm = LSTMModel(input_size, hidden_dim, num_layers, output_size, lag)
    cr = nn.MSELoss()
    import torch.optim as optim
    opt2 = optim.Adam(lstm.parameters(), lr)
    dataset = TensorDataset(x, samp_ts, y)
    concatenation_layer = ConcatenationLayer()

    if preload:
        vae.load_state_dict(torch.load(loadfile))

    combined_model = CombinedModel(vae, lstm, concatenation_layer, input_size , output_size)

    optim = torch.optim.Adam(combined_model.parameters(), betas=(0.9, 0.999), lr=0.0001)
    streaming_data_loader = StreamingDataLoader(dataset, batch_size,
                                                simulate_stream_speed=0.5)  # Adjust the speed as needed
    num_batches = x.shape[0]
    losses1 = []
    losses2 = []
    losses3 = []
    losses4 = []
    msev = []
    preds = []
    trues = []
    def cumavg(m):
        cumsum = np.cumsum(m)
        return cumsum / np.arange(1, cumsum.size + 1)


    start = time.time()
    for batch_idx in range(num_batches):
        xr, t, yr = next(streaming_data_loader)
        xr = xr.transpose(0, 1)
        t = t.transpose(0, 1)
        optim.zero_grad()
        concatenated_output, output1, output2, z, z_mean, z_log_var = combined_model(xr, t, yr)

        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
        l1 = 0.01 * torch.abs(z).sum(-1)
        lossL1 = 0.5 * ((yr - concatenated_output) ** 2).sum(-1).sum(0) + kl_loss / noise_std ** 2 + kl_loss + l1
        loss = 0.5 * ((yr - concatenated_output) ** 2).sum(-1).sum(0) / noise_std ** 2 + kl_loss

        lossL1 = torch.mean(lossL1)
        lossL1 /= lag

        loss = torch.mean(loss)
        loss /= lag
        MSEV = torch.mean((yr - concatenated_output) ** 2)
        calculate_memory_usage()
        loss4 = (MSEV + kl_loss) / lag
        if lossfun == 1:
            lossL1.backward()
        if lossfun == 2:
            loss.backward()
        if lossfun == 3:
            MSEV.backward()
        if lossfun == 4:
            loss4.backward()

        optim.step()
        losses1.append(lossL1.item())
        losses2.append(loss.item())
        losses3.append(MSEV.item())
        losses4.append(loss4.item())
        p = concatenated_output.reshape(output_size)
        yt = yr.reshape(output_size)
        print(f'Batch [{batch_idx + 1}], Loss: {loss.item():.4f},  mse: {MSEV.item():.4f}')
        outputp = f"pre3_{datasetnane}_{task}_{flag}_{hidden_dim}_{latent_dim}_{lr}_{lossfun}_{reguler}"
        with open(os.path.join(resultpath, f"{outputp}_Pred.csv"), 'a', newline='') as file:
            file.write(','.join(map(str, (p.detach().numpy()))) + '\n')
        with open(resultpath+outputp+"_True.csv", 'a', newline='') as file1:
            file1.write(','.join(map(str, (yt.detach().numpy()))) + '\n')
    end = time.time()
    exp_time = end-start
    print("Online testing finished!")
    print("Online testing time:", exp_time )
   # print(np.mean(losses1), np.mean(losses2), np.mean(losses3),np.mean(losses4))
    np.savetxt(resultpath+outputp+"_msev.csv", (msev))
    accu = cumavg(msev)
   # np.savetxt(resultpath+outputp+"_Accmsev.csv", (accu))

