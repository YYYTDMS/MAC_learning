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


with open('./data_m3/m3_col_r1.pkl', 'rb') as f80:
  LLM_pred_test = pickle.load(f80)




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
        nn.init.xavier_uniform_(self.level_embeddings.weight)  # Xavier 初始化

    def forward(self, input=None):
        embeddings_idx = [num for num in range(4880)]
        embeddings_idx = torch.tensor(embeddings_idx).to(device)

        embeddings = self.level_embeddings(embeddings_idx)

        return embeddings

class Encoder(nn.Module):
  def __init__(self, code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, personal_gate, hyperG_gate, PGatt_gate,X):
    super(Encoder, self).__init__()

    self.hier_embed_layer = HierarchicalEmbedding(X)

    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax()
    # Aggregate visit embeddings sequentially with attention
    self.temporal_edge_aggregator = nn.GRU(visit_dim, hdim, 1, batch_first=True)
    self.attention_context = nn.Linear(hdim, 1, bias=False)

  def forward(self, H, TE):
    #print(np.shape(X_G))
    #print(X_G)
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


class Final_Mlp(nn.Module):
    def __init__(self, hdim,nclass):
        super(Final_Mlp, self).__init__()
        self.final_layer = nn.Linear(nclass*2 , nclass)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,visit_emb,add_visit_emb):

        final_emb = torch.cat([visit_emb, add_visit_emb], dim=-1)

        output = self.softmax(self.final_layer(final_emb))
        return output

class ProHealth_VAE(nn.Module):
  def __init__(self, code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer,
               nhead, nclass, dist_dim, ode_hid, personal_gate,
               hyperG_gate, PGatt_gate, ODE_gate,disease_emb):
    super(ProHealth_VAE, self).__init__()
    self.encoder = Encoder(code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, personal_gate, hyperG_gate, PGatt_gate,disease_emb)
    self.ODE_Func = GRUODECell_Autonomous(hdim*1)
    self.decoder = ODE_VAE_Decoder(hdim, dist_dim, ode_hid, nclass, self.ODE_Func)
    self.softmax = nn.Softmax()
    self.criterion = nn.BCELoss()
    #self.visit_emb_pretrain=visit_emb
  def forward(self, Hs, TEs, timestamps, seq_lens,pids,duration_dict,past,truth,LLM_pred_vae):

    h = torch.stack([self.encoder(Hs[ii][:, 0:int(seq_lens[ii])], TEs[ii]) for ii in range(len(Hs))])

    mu = self.decoder.fc_mu(h)
    log_var = self.decoder.fc_var(h)
    z=self.decoder.reparameterize(mu, log_var)
    X = self.encoder.hier_embed_layer(None)
    zi=z
    pred_z = odeint(func = self.decoder.odefunc, y0 = zi, t = timestamps, method = 'rk4', options=dict(step_size=0.1))
    pred_z = torch.swapaxes(pred_z, 0, 1)
    temp_pred_z=[]
    last_visit_emb=[]

    for i, traj in enumerate(pred_z):
      duration = duration_dict[pids[i].item()]
      temp = [sum(duration[0:gap+1]) for gap in range(len(duration))]
      ts = [stamp / cur_max for stamp in temp]
      idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
      diag_idx = torch.tensor(LLM_pred_vae[i], dtype=torch.long, device=X.device)  # 确保在同一 device 上
      n = diag_idx.numel()

      mask = torch.zeros(4880, device=X.device, dtype=torch.float32)

      mask[diag_idx] = 1

      temp_pred_z.append(mask)

      last_visit_emb.append(traj[idx[-1],:])
    temp_pred_z=torch.stack(temp_pred_z)
    last_visit_emb=torch.stack(last_visit_emb)

    return last_visit_emb,temp_pred_z




import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


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
            r[i] += len(it) / len(t)
    return a / len(y_true_hot), r / len(y_true_hot)


import copy
import numpy as np
def code_level(labels, predicts):
    labels = np.array(labels)
    total_labels = np.where(labels == 1)[0].shape[0]
    top_ks = [10, 20]
    total_correct_preds = []
    for k in top_ks:
        correct_preds = 0
        for i, pred in enumerate(predicts):
            index = pred[:k]

            for ind in index:
                if labels[i][ind] == 1:
                    correct_preds = correct_preds + 1
        total_correct_preds.append(float(correct_preds))

    total_correct_preds = np.array(total_correct_preds) / total_labels
    return total_correct_preds


def visit_level(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10, 20]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            actual_positives = np.sum(labels[i])
            denominator = min(k, actual_positives)
            # top_k_indices = np.argsort(-predicts[i])[:k]
            top_k_indices = predicts[i][:k]
            true_positives = np.sum(labels[i][top_k_indices])
            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks




def train(model, lrate, num_epoch, train_loader, test_loader, model_directory, ode_gate, duration_dict, early_stop_range, balance, cur_max):

    init_model = copy.deepcopy(model)
    target_pred={}
    all_time=time.time()
    test_idx=0
    for (hyperGs, labels, pids, TEs, seq_lens) in test_loader:

      hyperGs = hyperGs.to(device); labels = labels.to(device); TEs = TEs.to(device)
      for patient_num in range(len(labels)):
        hyperGs_vae=[]
        labels_vae=[]
        TEs_vae=[]
        pids_vae=[]
        seq_lens_vae=[]
        LLM_pred_vae = []

        if seq_lens[patient_num] <= 1:continue
        if test_idx not in LLM_pred_test:
            test_idx+=1
            continue
        state = torch.load("./model/m3_5_STCare_nn_few_loss.pth", map_location="cpu")
        model.load_state_dict(state, strict=False)
        init_model.load_state_dict(state, strict=False)

        model.train()
        criterion = nn.BCELoss()
        losses = []
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.decoder.final_layer.parameters()),
                                     lr=lrate)
        patience=0
        patience_limit=50
        p1_list = []
        p2_list = []
        a1_list = []
        a2_list = []
        LLM_pred_vae.append(LLM_pred_test[test_idx])
        hyperGs_vae.append(hyperGs[patient_num])
        labels_vae.append(labels[patient_num])
        TEs_vae.append(TEs[patient_num])
        pids_vae.append(pids[patient_num])
        seq_lens_vae.append(seq_lens[patient_num])
        timestamps = []
        for pid in pids_vae:
          duration = duration_dict[pid.item()]
          timestamps += [sum(duration[0:gap+1]) for gap in range(len(duration))]
        temp = [stamp / cur_max for stamp in list(set(timestamps))]
        timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
        hyperGs_vae = torch.stack(hyperGs_vae).to(device)
        labels_vae = torch.stack(labels_vae).to(device)
        TEs_vae = torch.stack(TEs_vae).to(device)
        labels_vae = labels_vae.detach().cpu().numpy()
        print()
        start=time.time()
        print(f"====test {test_idx}=====")
        sample_epoch=0
        best_loss = float('inf')
        best_epoch=0
        while sample_epoch<50:
        # while patience<patience_limit:

            last_visit_emb,fake_label= model(hyperGs_vae, TEs_vae, timestamps, seq_lens_vae,pids_vae,duration_dict,hyperGs_vae,labels_vae,LLM_pred_vae)

            new_visit_emb=model.decoder.softmax(model.decoder.final_layer(last_visit_emb))
            original_visit_emb = init_model.decoder.softmax(init_model.decoder.final_layer(last_visit_emb))

            log_p = torch.log(new_visit_emb + 1e-12)
            q = original_visit_emb
            KL_loss = F.kl_div(log_p, q, reduction='batchmean')
            bce_loss=criterion(new_visit_emb,fake_label)

            loss = 0.1 * KL_loss + bce_loss


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()


            pred = new_visit_emb
            pred = torch.argsort(pred, dim=-1, descending=True)
            preds = pred.detach().cpu().numpy()

            best_state_dict = model.decoder.final_layer.state_dict()
            torch.save(best_state_dict,
                       f'{model_directory}/m3_patient_specific/sample{test_idx}.pth')
            target_pred[test_idx] = preds
            sample_epoch += 1

        test_idx += 1



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
train(model, 0.001/10, 1000, train_loader, test_loader, model_directory, True, patient_time_duration_encoded, 10, 0.5, cur_max)