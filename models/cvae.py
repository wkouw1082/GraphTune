"""
Conditional Variational AutoEncoder(CVAE) を定義するモジュール.
"""

import torch
from torch import nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import encoder, decoder


class CVAE(nn.Module):
    def __init__(self, dfs_size, time_size, node_size, edge_size, condition_size, params, device):
        super(CVAE, self).__init__()
        emb_size = params.model_params["emb_size"]
        en_hidden_size = params.model_params["en_hidden_size"]
        de_hidden_size = params.model_params["de_hidden_size"]
        self.rep_size = params.model_params["rep_size"]
        self.alpha = params.model_params["alpha"]
        self.beta = params.model_params["beta"]
        self.device = device
        self.encoder = encoder.Encoder(dfs_size, emb_size, en_hidden_size, self.rep_size, params, self.device)
        self.decoder = decoder.Decoder(self.rep_size, dfs_size, emb_size, de_hidden_size, time_size, node_size, edge_size, condition_size, params, self.device)

    def noise_generator(self, rep_size, batch_num):
        return torch.randn(batch_num, rep_size)

    def forward(self, x, word_drop=0):
        mu, sigma = self.encoder(x)
        z = transformation(mu, sigma, self.device)
        tu, tv, lu, lv, le = self.decoder(z, x)
        return mu, sigma, tu, tv, lu, lv, le

    def generate(self, data_num, conditional_label, max_size, z=None, is_output_sampling=True):
        if z is None:
            z = self.noise_generator(self.rep_size, data_num).unsqueeze(1)
            z = z.to(self.device)
        tu, tv, lu, lv, le =\
            self.decoder.generate(z, conditional_label, max_size, is_output_sampling)
        return tu, tv, lu, lv, le

    def loss(self, encoder_loss, decoder_loss):
        """CVAEの損失関数

        Args:
            encoder_loss (_type_): Encoder modelのloss
            decoder_loss (_type_): Decoder modelのloss

        Returns:
            (): CVAEのloss
        """
        cvae_loss = self.beta * encoder_loss + self.alpha * decoder_loss
        return cvae_loss


def transformation(mu, sigma, device):
    """Reparametrization trick
    
    mu, sigma, 正規分布から取得したノイズから潜在変数zを計算する.

    Args:
        mu (_type_): _description_
        sigma (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    return mu + torch.exp(0.5*sigma) * torch.randn(sigma.shape).to(device)


if __name__ == "__main__":
    print("vae.py")