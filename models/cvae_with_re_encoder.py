"""
Conditional Variational AutoEncoder(CVAE) + ReEncoderモデルのインターフェースを定義するモジュール.
"""

import torch
from torch import nn
import sys
import os
import torch.nn.functional as F
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import try_gpu, sample_dist, convert2onehot
from models.re_encoder import ReEncoder
from models.cvae import CVAE


class CVAEwithReEncoder(nn.Module):
    """Conditional VAE + ReEncoderのインターフェース

    input_data => CVAE(Encoder, Decoder) => output_data => ReEncdoer() => predicted_graph_property
    """
    def __init__(self, dfs_size, time_size, node_size, edge_size, condition_size, params, device):
        super(CVAEwithReEncoder, self).__init__()
        self.gamma = params.model_params['gamma']
        self.device = device
        self.cvae = CVAE(dfs_size, time_size, node_size, edge_size, condition_size, params, device)
        self.re_encoder = ReEncoder(dfs_size-1, params, device)

    def forward(self, x):
        # CVAE
        mu, sigma, tu, tv, lu, lv, le = self.cvae(x)
        # ReEncoder
        graph_property = self.re_encoder(torch.cat([tu, tv, lu, lv, le], dim=2))
        return mu, sigma, tu, tv, lu, lv, le, graph_property

    def loss(self, encoder_loss, decoder_loss, re_encoder_loss):
        """Loss function

        Args:
            encoder_loss (_type_): Encoder modelのloss
            decoder_loss (_type_): Decoder modelのloss
            re_decoder_loss (_type_): ReEncoder modelのloss

        Returns:
            (): CVAEwithReEncoderのloss
        """
        cvae_loss = self.cvae.loss(encoder_loss, decoder_loss) + self.gamma * re_encoder_loss
        return cvae_loss

    def generate(self, data_num, conditional_label, max_size, z=None, is_output_sampling=True):
        """Generate graph samples

        Args:
            data_num                   (int): 生成サンプル数
            conditional_label (torch.Tensor): 条件として与えるラベル情報
            max_size                   (int): 最大エッジ数
            z                 (torch.Tensor): 潜在空間からサンプリングされたデータ
            is_output_sampling        (bool): Trueなら返り値を予測dfsコードからargmaxしたものに. Falseなら予測分布を返す

        Returns:
            (torch.Tensor): 生成されたサンプルの5-tuplesの各要素のデータ
        """
        tu, tv, lu, lv, le = self.cvae.generate(data_num, conditional_label, max_size, z, is_output_sampling)
        return tu, tv, lu, lv, le



if __name__ == "__main__":
    print("cvae_with_re_encoder.py")
    import config
    params = config.Parameters()
    model = CVAEwithReEncoder(dfs_size=173, time_size=51, node_size=34, edge_size=2, condition_size=1, params=params, device="cuda")
    print(model)