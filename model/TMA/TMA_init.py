import time
import math
import pickle as pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint
import random
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0")



def load(emb_file_path):

    emb_dict = {}
    with open(emb_file_path,'r') as emb_file:        
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                emb_dict[index] = np.array(emb.split()).astype(np.float32)
        
    return train_para, emb_dict  


disease_emb = np.load('./data_m3/m3_diag_emb.npy')
disease_emb = torch.from_numpy(disease_emb).to(device)

with open('./data_m3/binary_train_codes_x.pkl', 'rb') as f0:
  binary_train_codes_x = pickle.load(f0)

with open('./data_m3/binary_test_codes_x.pkl', 'rb') as f1:
  binary_test_codes_x = pickle.load(f1)

train_codes_y = np.load('./data_m3/train_codes_y.npy')
train_visit_lens = np.load('./data_m3/train_visit_lens.npy')

test_codes_y = np.load('./data_m3/test_codes_y.npy')
test_visit_lens = np.load('./data_m3/test_visit_lens.npy')
train_pids = np.load('./data_m3/train_pids.npy')

test_pids = np.load('./data_m3/test_pids.npy')


with open('./data_m3/patient_time_duration_encoded.pkl', 'rb') as f80:
  patient_time_duration_encoded = pickle.load(f80)



def transform_and_pad_input(x):
  tempX = []
  for ele in x:
    tempX.append(torch.tensor(ele).to(torch.float32))
  x_padded = pad_sequence(tempX, batch_first=True, padding_value=0)
  return x_padded

trans_y_train = torch.tensor(train_codes_y)
trans_y_test = torch.tensor(test_codes_y)
padded_X_train = torch.transpose(transform_and_pad_input(binary_train_codes_x), 1, 2)
padded_X_test = torch.transpose(transform_and_pad_input(binary_test_codes_x), 1, 2)
class_num = train_codes_y.shape[1]

total_pids = list(train_pids) + list(test_pids)
cur_max = 0
for pid in total_pids:
  duration = patient_time_duration_encoded[pid]
  ts = [sum(duration[0:gap+1]) for gap in range(len(duration))]
  if cur_max < max(ts):
    cur_max = max(ts)
class ProHealth_Dataset(data.Dataset):
    def __init__(self, hyperG, data_label, pid, duration_dict, data_len, te_location, Dv):
        self.hyperG = hyperG
        self.data_label = data_label
        self.pid = pid
        self.data_len = data_len
        if te_location == None:
          TE_list = [prepare_temporal_encoding(hyperG[j], pid[j], duration_dict, Dv) for j in range(len(hyperG))]
          self.TE = pad_sequence(TE_list, batch_first=True, padding_value=0)
        else:
            pass
 
    def __len__(self):
        return len(self.hyperG)
 
    def __getitem__(self, idx):
        return self.hyperG[idx], self.data_label[idx], self.pid[idx], self.TE[idx], self.data_len[idx]



# In[ ]:


def glorot(tensor):
  if tensor is not None:
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    tensor.data.uniform_(-stdv, stdv)

class MLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=2048, output_dim=128):
        super(MLP, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projector(x)

class HierarchicalEmbedding(nn.Module):
    def __init__(self, embeddings):
        super(HierarchicalEmbedding, self).__init__()
        num_embeddings, embedding_dim = embeddings.shape

        self.level_embeddings = nn.Embedding(num_embeddings, 128)
        nn.init.xavier_uniform_(self.level_embeddings.weight)

    def forward(self, input=None):
        embeddings_idx = [num for num in range(4880)]
        embeddings_idx = torch.tensor(embeddings_idx).to(device)
        # print(embeddings_idx)
        embeddings = self.level_embeddings(embeddings_idx)
        # print(embeddings)
        return embeddings  #

class Encoder(nn.Module):
  def __init__(self, code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, personal_gate, hyperG_gate, PGatt_gate,X):
    super(Encoder, self).__init__()

    self.hier_embed_layer = HierarchicalEmbedding(X)

    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax()

    self.temporal_edge_aggregator = nn.GRU(visit_dim, hdim, 1, batch_first=True)
    self.attention_context = nn.Linear(hdim, 1, bias=False)

  def forward(self, H, TE):
    X=self.hier_embed_layer(None)
    visit_emb = torch.matmul(H.T.to(torch.float32), X)
    hidden_states, _ = self.temporal_edge_aggregator(visit_emb)
    alpha1 = self.softmax(torch.squeeze(self.attention_context(hidden_states), 1))
    h = torch.sum(torch.matmul(torch.diag(alpha1), hidden_states), 0)
    return h


class GRUODECell_Autonomous(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias        = bias

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh

class ODEFunc(nn.Module):
  def __init__(self, hdim, ode_hid):
    super().__init__()
    self.func = nn.Sequential(nn.Linear(hdim, ode_hid),
                              nn.Tanh(),
                              nn.Linear(ode_hid, hdim))
    for m in self.func.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)
        nn.init.constant_(m.bias, val=0)

  def forward(self, t, y):
    output = self.func(y)
    return output

class ODE_VAE_Decoder(nn.Module):
  def __init__(self, hdim, dist_dim, ode_hid, nclass, ODE_Func):
    super(ODE_VAE_Decoder, self).__init__()
    self.fc_mu = nn.Linear(hdim, dist_dim)
    self.fc_var = nn.Linear(hdim, dist_dim)
    self.fc_mu0 = nn.Linear(hdim, dist_dim)
    self.fc_var0 = nn.Linear(hdim, dist_dim)
    self.relu = nn.ReLU()
    self.odefunc = ODE_Func
    # self.final_layer = nn.Linear(hdim*3, nclass)
    self.final_layer = nn.Linear(hdim * 1, nclass)
    self.softmax = nn.Softmax(dim=-1)
  def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    q = torch.distributions.Normal(mu, std)
    return q.rsample()

  def forward(self,z,timestamps):
    pred_z = odeint(func = self.odefunc, y0 = z, t = timestamps, method = 'rk4', options=dict(step_size=0.1))
    output = self.softmax(self.final_layer(pred_z))
    return output


class ProHealth_VAE(nn.Module):
  def __init__(self, code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, nclass, dist_dim, ode_hid, personal_gate, hyperG_gate, PGatt_gate, ODE_gate,disease_emb):
    super(ProHealth_VAE, self).__init__()
    self.encoder = Encoder(code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, personal_gate, hyperG_gate, PGatt_gate,disease_emb)
    self.ODE_Func = GRUODECell_Autonomous(hdim*1)
    self.decoder = ODE_VAE_Decoder(hdim, dist_dim, ode_hid, nclass, self.ODE_Func)
    self.softmax = nn.Softmax()


  def forward(self, Hs, TEs, timestamps, seq_lens,pids,duration_dict,past,truth):

    h = torch.stack([self.encoder(Hs[ii][:, 0:int(seq_lens[ii])], TEs[ii]) for ii in range(len(Hs))])

    mu = self.decoder.fc_mu(h)
    log_var = self.decoder.fc_var(h)
    z=self.decoder.reparameterize(mu, log_var)

    zi=z
    pred_z = odeint(func = self.decoder.odefunc, y0 = zi, t = timestamps, method = 'rk4', options=dict(step_size=0.1))


    pred2=self.decoder.softmax(self.decoder.final_layer(pred_z))

    pred2 = torch.swapaxes(pred2, 0, 1)
    
    mug=mu
    log_varg=log_var

    ELBO = torch.mean(-0.5 * torch.sum(1 + log_varg - mug ** 2 - log_varg.exp(), dim=1))

    return 0,0,pred2,mug,log_varg,ELBO

  def predict(self,Hs, TEs, timestamps, seq_lens,pids,duration_dict):
    h = torch.stack([self.encoder(Hs[ii][:, 0:int(seq_lens[ii])], TEs[ii]) for ii in range(len(Hs))])


    diff_loss=0

    mu = self.decoder.fc_mu(h)
    log_var = self.decoder.fc_var(h)
    z=self.decoder.reparameterize(mu, log_var)

    zi = z

    pred2=self.decoder(zi,timestamps)
    pred2 = torch.swapaxes(pred2, 0, 1)
    return diff_loss,pred2



def ProHealth_loss(pred, truth, past, pids, mu, log_var, duration_dict, timestamps, ode_gate, balance, cur_max,ELBO):
  criterion = nn.BCELoss()
  if not ode_gate:
    loss = criterion(pred, truth)
  else:
    reconstruct_loss = 0
    last_visits = []
    for i, traj in enumerate(pred):
      duration = duration_dict[pids[i].item()]
      temp = [sum(duration[0:gap+1]) for gap in range(len(duration))]
      ts = [stamp / cur_max for stamp in temp]
      idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
      visit_lens = len(ts)
      last_visits.append(traj[idx[-1], :])
      reconstruct_loss += criterion(traj[idx[:-1], :], torch.swapaxes(past[i][:, 0:(visit_lens - 1)], 0, 1))

    last_visits = torch.stack(last_visits)
    reconstruct_loss = (reconstruct_loss / len(pred))
    pred_loss = criterion(last_visits, truth)
    loss =  pred_loss + balance * ELBO +reconstruct_loss
  return loss

import torch
import numpy as np
from sklearn.metrics import f1_score


def f1(y_true_hot, y_pred):
    result = np.zeros_like(y_true_hot)#[]
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1

    return f1_score(y_true=y_true_hot, y_pred=result, average='weighted', zero_division=0)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            # r[i] += len(it) / min(k, len(t))
            r[i] += len(it) / len(t)
    return a / len(y_true_hot), r / len(y_true_hot)


def train(model, lrate, num_epoch, train_loader, test_loader, model_directory, ode_gate, duration_dict, early_stop_range, balance, cur_max):
  model.train()
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lrate)
  test_loss_per_epoch = []; train_average_loss_per_epoch = []
  p1_list = []; p2_list = []; p3_list = []; p4_list = []; p5_list = []; p6_list = []
  r1_list = []; r2_list = []; r3_list = []; r4_list = []; r5_list = []; r6_list = []
  n1_list = []; n2_list = []; n3_list = []; n4_list = []; n5_list = []; n6_list = []
  for epoch in range(num_epoch):
      start = time.time()
      one_epoch_train_loss = []
      train_idx = 0
      for i, (hyperGs, labels, pids, TEs, seq_lens) in enumerate(train_loader):
          hyperGs = hyperGs.to(device);
          labels = labels.to(device);
          TEs = TEs.to(device)
          hyperGs_vae = []
          labels_vae = []
          TEs_vae = []
          pids_vae = []
          seq_lens_vae = []

          for patient_num in range(len(labels)):
              if seq_lens[patient_num] > 1:
                  hyperGs_vae.append(hyperGs[patient_num])
                  labels_vae.append(labels[patient_num])
                  TEs_vae.append(TEs[patient_num])
                  pids_vae.append(pids[patient_num])
                  seq_lens_vae.append(seq_lens[patient_num])

          hyperGs_vae = torch.stack(hyperGs_vae).to(device)
          labels_vae = torch.stack(labels_vae).to(device)
          TEs_vae = torch.stack(TEs_vae).to(device)
          timestamps = []
          for pid in pids_vae:
              duration = duration_dict[pid.item()]
              timestamps += [sum(duration[0:gap + 1]) for gap in range(len(duration))]
          temp = [stamp / cur_max for stamp in list(set(timestamps))]
          timestamps = torch.tensor(temp).to(torch.float32).sort()[0]

          _, pred1, pred2, mu, log_var, ELBO = model(hyperGs_vae, TEs_vae, timestamps, seq_lens_vae, pids_vae,
                                                         duration_dict, hyperGs_vae,
                                                         labels_vae.to(torch.float32))
          loss2 = ProHealth_loss(pred2, labels_vae.to(torch.float32), hyperGs_vae, pids_vae, mu, log_var, duration_dict,
                                 timestamps, ode_gate, 0.05, cur_max, ELBO)

          loss = loss2
          one_epoch_train_loss.append(loss.item())
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
          optimizer.step()
      end = time.time() - start

      train_average_loss_per_epoch.append(sum(one_epoch_train_loss) / len(one_epoch_train_loss))
      print('Epoch: [{}/{}], Average Loss: {}'.format(epoch + 1, num_epoch, round(train_average_loss_per_epoch[-1], 9)))

      model.eval()
      one_epoch_test_loss = []
      test_data_len = 0
      pred_list = []
      truth_list = []
      temp_test_loss_per_epoch = []
      for (hyperGs, labels, pids, TEs, seq_lens) in test_loader:
          hyperGs = hyperGs.to(device);
          labels = labels.to(device);
          TEs = TEs.to(device)
          hyperGs_vae = []
          labels_vae = []
          TEs_vae = []
          pids_vae = []
          seq_lens_vae = []
          for patient_num in range(len(labels)):
              if seq_lens[patient_num] > 1:
                  hyperGs_vae.append(hyperGs[patient_num])
                  labels_vae.append(labels[patient_num])
                  TEs_vae.append(TEs[patient_num])
                  pids_vae.append(pids[patient_num])
                  seq_lens_vae.append(seq_lens[patient_num])
          hyperGs_vae = torch.stack(hyperGs_vae).to(device)
          labels_vae = torch.stack(labels_vae).to(device)
          TEs_vae = torch.stack(TEs_vae).to(device)
          with torch.no_grad():
              timestamps = []
              for pid in pids_vae:
                  duration = duration_dict[pid.item()]
                  timestamps += [sum(duration[0:gap + 1]) for gap in range(len(duration))]
              temp = [stamp / cur_max for stamp in list(set(timestamps))]
              timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
              test_diff_loss, pred2 = model.predict(hyperGs_vae, TEs_vae, timestamps, seq_lens_vae, pids_vae,
                                                    duration_dict)
              test_loss2 = ProHealth_loss(pred2, labels_vae.to(torch.float32), hyperGs_vae, pids_vae, 0, 0,
                                          duration_dict,
                                          timestamps, ode_gate, 0, cur_max, 0)  # pred+ELBO+supervision
              test_loss = test_diff_loss * 0.01 + test_loss2
              temp_test_loss_per_epoch.append(test_loss.item())
          test_data_len += len(pids_vae)
          truth_list.append(labels_vae)
          traj_pred_list = []

          if ode_gate:
              for jj, traj in enumerate(pred2):
                  duration = duration_dict[pids_vae[jj].item()]
                  ts1 = [sum(duration[0:gap + 1]) for gap in range(len(duration))]
                  ts = [stamp / cur_max for stamp in ts1]
                  idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
                  pred_list.append(traj[idx[-1], :])
                  traj_pred_list.append(traj[idx, :].cpu().numpy())
          else:
              pred_list.append(pred)
      test_loss_per_epoch.append(sum(temp_test_loss_per_epoch))
      pred = torch.vstack(pred_list)

      truth = torch.vstack(truth_list)

      pred = torch.argsort(pred, dim=-1, descending=True)
      preds = pred.detach().cpu().numpy()

      labels = truth.detach().cpu().numpy()
      f1_score1 = f1(labels, preds)
      prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
      r1_list.append(recall[0]);
      r2_list.append(recall[1]);
      r3_list.append(recall[2]);
      r4_list.append(recall[3])
      n1_list.append(f1_score1);
      print("cur:", "Recall@10", r1_list[-1], "Recall@20:", r2_list[-1], "F1:", n1_list[-1])
      with open("m3_100_nn_test_traj_pred_list.pkl", "wb") as f:
          pickle.dump(traj_pred_list, f)

      model.train()


def prepare_temporal_encoding(H, pid, duration_dict, Dv):
  TE = []
  X_i_idx = torch.unique(torch.nonzero(H, as_tuple=True)[0])
  H_i = H[X_i_idx, :]
  for code in X_i_idx:
    TE_code = torch.zeros(Dv * 2)
    visits = torch.nonzero(H[code.item()]).tolist()
    temp = duration_dict[pid][:-1]
    code_duration = [sum(temp[0:gap+1]) for gap in range(len(temp))]
    visits.append([len(code_duration) - 1])
    pre_delta = [code_duration[visits[j][0]] - code_duration[visits[j - 1][0]] for j in range(1, len(visits))]
    delta = sum(pre_delta) / len(pre_delta)
    T_m = sum(code_duration)
    if T_m == 0:
      T_m += 1
    for k in range(len(TE_code)):
      if k < Dv:
        TE_code[k] = math.sin((k * delta) / (T_m * Dv))
      else:
        TE_code[k] = math.cos(((k - Dv) * delta) / (T_m * Dv))
    TE.append(TE_code)
  TE_i = torch.stack(TE)
  return TE_i
model = ProHealth_VAE(0, 32, 128, 128, 8, 64, 128,
                      2, 8, class_num, 128, 128,
                      False,False, False, True,disease_emb).to(device)
te_directory = None
training_data = ProHealth_Dataset(padded_X_train, trans_y_train, train_pids, patient_time_duration_encoded, train_visit_lens, te_directory, 32)
train_loader = DataLoader(training_data, batch_size=128, shuffle=False)
test_data = ProHealth_Dataset(padded_X_test, trans_y_test, test_pids, patient_time_duration_encoded, test_visit_lens, te_directory, 32)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
model_directory = './model'
r1_list, r2_list, n1_list= train(model, 0.0001/5, 500, train_loader, test_loader, model_directory, True, patient_time_duration_encoded, 10, 0.5, cur_max)