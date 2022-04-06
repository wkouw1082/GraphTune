"""
Encoder を定義するモジュール.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    """
    Encoderクラス.
    
    線形層1(input_size, emb_size) => LSTM(emb_size, hidden_size) => 線形層2(hidden_size, rep_size)
    """
    
    def __init__(self, input_size, emb_size, hidden_size, rep_size, params, device, num_layer=2):
        """Encoderのハイパーパラメータを設定する.

        Args:
            input_size  (int) : 入力データのサイズ
            emb_size    (int) : 埋め込み層への入力サイズ
            hidden_size (int) : LSTMの隠れ層のサイズ
            rep_size    (int) : LSTMの後に配置される線形層の出力サイズ
            device      ()    : デバイス情報
            num_layer (int, optional): LSTMの層数. Defaults to 2.
        """
        super(Encoder, self).__init__()
        self.emb = nn.Linear(input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True)
        self.mu = nn.Linear(hidden_size, rep_size)
        self.sigma = nn.Linear(hidden_size, rep_size)
        self.device = device
        self.num_layer = num_layer
        self.hidden_size = hidden_size

    def forward(self, x):
        """順伝搬

        Args:
            x (_type_): DFSコードのミニバッチ

        Returns:
            () : 分布の平均値に該当する値
            () : 分布の標準偏差に該当する値
        """
        # TODO 活性化関数を導入したほうが良い？
        x = self.emb(x)
        # x = torch.cat((x,label),dim=2)
        x, (h,c) = self.lstm(x)
        x = x[:, -1, :].unsqueeze(1)
        return self.mu(x), self.sigma(x)
    
    def loss(self, mu, sigma):
        """正規分布とのKL divergenceを計算する関数

        Args:
            mu    () : 分布の平均値に該当する値
            sigma () : 分布の分散に該当する値
        
        Returns:
            () : KL divergence
        """
        delta = 1e-7
        return -0.5 * torch.sum( 1 + sigma - mu**2 - torch.exp(sigma + delta))