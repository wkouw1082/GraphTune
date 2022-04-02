"""
Decoder を定義するモジュール.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import try_gpu, sample_dist, convert2onehot


class Decoder(nn.Module):
    """
    Decoder クラス.
    
    () => () => () => () => ()
    """
    
    def __init__(self, rep_size, input_size, emb_size, hidden_size, time_size, node_label_size, edge_label_size, condition_size, params, device, num_layer=3):
        """Decoderのハイパーパラメータを設定する.

        Args:
            rep_size (_type_): _description_
            input_size (_type_): _description_
            emb_size (_type_): _description_
            hidden_size (_type_): _description_
            time_size (_type_): _description_
            node_label_size (_type_): _description_
            edge_label_size (_type_): _description_
            condition_size (_type_): _description_
            params (_type_): configで設定されたglobal変数のset
            device (_type_): _description_
            num_layer (int, optional): _description_. Defaults to 3.
        """
        super(Decoder, self).__init__()
        self.sampling_generation = params.sampling_generation
        self.condition_size = condition_size
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.emb = nn.Linear(input_size, emb_size)
        # onehot vectorではなく連続値なためサイズは+2
        self.f_rep = nn.Linear(rep_size+condition_size, input_size)
        self.lstm = nn.LSTM(emb_size+rep_size+condition_size, hidden_size, num_layers=self.num_layer, batch_first=True)
        self.f_tu = nn.Linear(hidden_size, time_size)
        self.f_tv = nn.Linear(hidden_size, time_size)
        self.f_lu = nn.Linear(hidden_size, node_label_size)
        self.f_lv = nn.Linear(hidden_size, node_label_size)
        self.f_le = nn.Linear(hidden_size, edge_label_size)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.5)

        self.f_h = nn.Linear(hidden_size, hidden_size)
        self.f_c = nn.Linear(hidden_size, hidden_size)

        self.time_size = time_size
        self.node_label_size = node_label_size
        self.edge_label_size = edge_label_size

        self.ignore_label = params.ignore_label
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction="sum")
        self.device = device

    def forward(self, rep, x, word_drop=0):
        """
        学習時のforward
        Args:
            rep: encoderの出力
            x: dfs code
        Returns:
            tu: source time
            tv: sink time
            lu: source node label
            lv: sink node label
            le: edge label
        """
        conditional =x[:, 0, -1*self.condition_size:].unsqueeze(1)
        
        rep = torch.cat([rep, conditional], dim=2)

        origin_rep=rep
        rep = self.f_rep(rep)
        #rep = self.dropout(rep)

        x = torch.cat((rep, x), dim=1)[:,:-1,:]
        h_0 = try_gpu(self.device, torch.Tensor())
        c_0 = try_gpu(self.device, torch.Tensor())

        batch_size = x.shape[0]

        for batch in range(x.shape[0]):
            conditional_value = x[batch, 0, -1*self.condition_size].item()
            h_0 = torch.cat((h_0, try_gpu(self.device, torch.Tensor(self.num_layer, 1, self.hidden_size).fill_(conditional_value))), dim=1)
            c_0 = torch.cat((c_0, try_gpu(self.device, torch.Tensor(self.num_layer, 1, self.hidden_size).fill_(conditional_value))), dim=1)

        # word drop
        for batch in range(x.shape[0]):
            args=random.choices([i for i in range(x.shape[1])], k=int(x.shape[1]*word_drop))
            zero=try_gpu(self.device, torch.zeros([1, 1, x.shape[2] - self.condition_size]))
            x[batch,args, :-1 * self.condition_size]=zero
        x = self.emb(x)
        rep = torch.cat([origin_rep for _ in range(x.shape[1])],dim=1)
        x = torch.cat((x,rep),dim=2)

        # h_0 = self.f_h(h_0)
        # h_0 = F.relu(h_0)
        # c_0 = self.f_c(c_0)
        # c_0 = F.relu(c_0)

        # h_0 = try_gpu(self.device, torch.Tensor(self.num_layer, batch_size, self.hidden_size).fill_(conditional_value))
        # c_0 = try_gpu(self.device, torch.Tensor(self.num_layer, batch_size, self.hidden_size).fill_(conditional_value))

        # h_0 = try_gpu(self.device, torch.zeros((self.num_layer, batch_size, self.hidden_size)))
        # c_0 = try_gpu(self.device, torch.zeros((self.num_layer, batch_size, self.hidden_size)))

        x, (h, c) = self.lstm(x, (h_0,c_0))
        x = self.dropout(x)
        
        tu = self.softmax(self.f_tu(x))
        tv = self.softmax(self.f_tv(x))
        lu = self.softmax(self.f_lu(x))
        lv = self.softmax(self.f_lv(x))
        le = self.softmax(self.f_le(x))
        return tu, tv, lu, lv, le

    def generate(self, rep, conditional_label, max_size=100, is_output_sampling=True):
        """
        生成時のforward. 生成したdfsコードを用いて、新たなコードを生成していく
        Args:
            rep: encoderの出力
            max_size: 生成を続ける最大サイズ(生成を続けるエッジの最大数)
            is_output_sampling: Trueなら返り値を予測dfsコードからargmaxしたものに. Falseなら予測分布を返す
        Returns:
        """
        conditional_value = conditional_label.item()
        conditional_label = conditional_label.unsqueeze(0).unsqueeze(1)
        conditional_label = torch.cat([conditional_label for _ in range(rep.shape[0])], dim=0)
        conditional_label = try_gpu(self.device,conditional_label)

        rep = torch.cat([rep, conditional_label], dim=2)

        origin_rep=rep

        rep = self.f_rep(rep)
        rep = self.emb(rep)
        x = rep
        x = torch.cat((x,origin_rep),dim=2)
        batch_size = x.shape[0]

        tus = torch.LongTensor()
        tus = try_gpu(self.device,tus)
        tvs = torch.LongTensor()
        tvs = try_gpu(self.device,tvs)
        lus = torch.LongTensor()
        lus = try_gpu(self.device,lus)
        lvs = torch.LongTensor()
        lvs = try_gpu(self.device,lvs)
        les = torch.LongTensor()
        les = try_gpu(self.device,les)

        tus_dist=try_gpu(self.device,torch.Tensor())
        tvs_dist=try_gpu(self.device,torch.Tensor())
        lus_dist=try_gpu(self.device,torch.Tensor())
        lvs_dist=try_gpu(self.device,torch.Tensor())
        les_dist=try_gpu(self.device,torch.Tensor())

        h_0 = try_gpu(self.device, torch.Tensor(self.num_layer, batch_size, self.hidden_size).fill_(conditional_value))
        c_0 = try_gpu(self.device, torch.Tensor(self.num_layer, batch_size, self.hidden_size).fill_(conditional_value))

        # h_0 = self.f_h(h_0)
        # h_0 = F.relu(h_0)
        # c_0 = self.f_c(c_0)
        # c_0 = F.relu(c_0)

        # h_0 = try_gpu(self.device, torch.zeros((self.num_layer, batch_size, self.hidden_size)))
        # c_0 = try_gpu(self.device, torch.zeros((self.num_layer, batch_size, self.hidden_size)))

        for i in range(max_size):
            if i == 0:
                x, (h, c) = self.lstm(x, (h_0,c_0))
                # x, (h, c) = self.lstm(x)
            else:
                x = self.emb(x)
                x = torch.cat((x,origin_rep),dim=2)
                x, (h, c) = self.lstm(x, (h, c))

            tu_dist = self.softmax(self.f_tu(x))
            tv_dist = self.softmax(self.f_tv(x))
            lu_dist = self.softmax(self.f_lu(x))
            lv_dist = self.softmax(self.f_lv(x))
            le_dist = self.softmax(self.f_le(x))
            
            tus_dist = torch.cat([tu_dist], dim=1)
            tvs_dist = torch.cat([tv_dist], dim=1)
            lus_dist = torch.cat([lu_dist], dim=1)
            lvs_dist = torch.cat([lv_dist], dim=1)
            les_dist = torch.cat([le_dist], dim=1)

            if self.sampling_generation:
                tu = sample_dist(tu_dist)  # サンプリングで次のエッジを決める
                tv = sample_dist(tv_dist)
                lu = sample_dist(lu_dist)
                lv = sample_dist(lv_dist)
                le = sample_dist(le_dist)
            else:
                tu = torch.argmax(tu_dist, dim=2)  # 最大値で次のエッジを決める
                tv = torch.argmax(tv_dist, dim=2)
                lu = torch.argmax(lu_dist, dim=2)
                lv = torch.argmax(lv_dist, dim=2)
                le = torch.argmax(le_dist, dim=2)

            tus = torch.cat((tus, tu), dim=1)
            tvs = torch.cat((tvs, tv), dim=1)
            lus = torch.cat((lus, lu), dim=1)
            lvs = torch.cat((lvs, lv), dim=1)
            les = torch.cat((les, le), dim=1)
            
            tu = tu.squeeze().cpu().detach().numpy()
            tv = tv.squeeze().cpu().detach().numpy()
            lu = lu.squeeze().cpu().detach().numpy()
            lv = lv.squeeze().cpu().detach().numpy()
            le = le.squeeze().cpu().detach().numpy()

            tu = convert2onehot(tu, self.time_size)
            tv = convert2onehot(tv, self.time_size)
            lu = convert2onehot(lu, self.node_label_size)
            lv = convert2onehot(lv, self.node_label_size)
            le = convert2onehot(le, self.edge_label_size)
            x = torch.cat((tu, tv, lu, lv, le), dim=1).unsqueeze(1)
            x = try_gpu(self.device,x)
    
            x = torch.cat((x, conditional_label),dim=2)
        if is_output_sampling:
            return tus, tvs, lus, lvs, les
        else:
            return tus_dist, tvs_dist, lus_dist, lvs_dist, les_dist
    
    def loss(self, results, targets):
        """Cross Entropyを計算する関数

        Args:
            results   (dict): Decoderから出力されたDFSコードの分布
            targets   (dict): labelデータ
        
        Returns:
            (dict): 5-tuplesの各要素のlossを持つdict
            (): 5-tuplesの各要素のlossのsum
        """
        total_loss = 0
        loss_dict = {}
        for i, (key, pred) in enumerate(results.items()):
            loss_dict[key] = self.criterion(pred.transpose(2, 1), targets[key])
            total_loss += loss_dict[key]
        return loss_dict.copy(), total_loss
    
    def accuracy(self, results, targets):
        """分類精度を計算する関数
        
        確率分布からargmaxを取ることでサンプリングする.

        Args:
            results (_type_): _description_
            targets (_type_): _description_

        Returns:
            (dict): 5-tuplesの各要素の分類精度
        """
        acc_dict = {}
        for i, (key, pred) in enumerate(results.items()):
            pred = torch.argmax(pred, dim=2)    # onehot => label
            pred = pred.view(-1)
            targets[key] = targets[key].view(-1)
            # 分類精度を計算
            score = torch.zeros(pred.shape[0])
            score[pred==targets[key]] = 1
            data_len = pred.shape[0]
            if not self.ignore_label is None:
                targets[key] = targets[key].cpu()
                ignore_args = np.where(targets[key]==self.ignore_label)[0]
                data_len -= len(ignore_args)
                score[ignore_args] = 0
            score = torch.sum(score)/data_len
            acc_dict[key] = score
        return acc_dict.copy()