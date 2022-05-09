"""
(C)VAEのDecoderから出力されたDFSコードの確率分布から、グラフ特徴量を算出するReEncoderモデルを定義するモジュール.
"""

import torch
from torch import nn


class ReEncoder(nn.Module):
    """Decoderの後ろに配置されるLSTM

    Decoderの後ろに配置されるLSTMであり、処理はEncoderと概ね同じである。
    DecodeされたDFSコードからグラフ特徴量を計算する。
    (モデル概要)
    線形層1(input_size, emb_sizae) => LSTM(emb_size, hidden_size) => 線形層2(hidden_size, rep_size)

    Functions
    ---------
    __init__() : LSTMとそのパラメータを定義する。
    forward()  : 順伝搬処理する。
    """
    def __init__(self, input_size, params, device, num_layer=2):
        """コンストラクタ

        Args
        ----
        input_size  (int) : LSTMの前に配置される線形層へ入力するデータの次元数
        params      (config.Parameters) : configのglobal変数のset
        device      (int) : cpu or cuda device
        num_layer   (int) : LSTMの層の数
        """
        super(ReEncoder, self).__init__()
        # 線形層1
        self.emb = nn.Linear(input_size, params.model_params["emb_size"])
        # LSTM
        self.lstm = nn.LSTM(params.model_params["emb_size"], params.model_params["re_en_hidden_size"], num_layers=num_layer, batch_first=True)
        # 線形層2
        self.calc_graph_property = nn.Linear(params.model_params["re_en_hidden_size"], params.model_params["re_en_rep_size"])
        # Loss function
        self.criterion = nn.MSELoss(reduction="sum")
        # Device info
        self.device = device
        # Acuuracy range
        self.acc_range = params.acc_range
        
    def forward(self, dfs_codes):
        """forwarding

        Args:
            dfs_codes (torch.tensor[batch_size, max_sequence_length, dfs_size-1]): onehotのDFSコードが格納されたミニバッチ

        Returns:
            (torch.tensor[rep_size]): ReEncoderのoutput
        """
        embedded_dfs_codes = self.emb(dfs_codes)
        # x = torch.cat((x,label),dim=2)
        output, (h,c) = self.lstm(embedded_dfs_codes)
        output = output[:, -1, :].unsqueeze(1)
        
        return self.calc_graph_property(output)

    def loss(self, results, targets):
        """ReEnocderの出力と入力グラフの統計量との間でMSE lossを計算する関数.

        Args:
            results (torch.Tensor): Reencoderの出力
            targets (torch.Tensor): 入力グラフの統計量

        Returns:
            (torch.Tensor): MSE loss
        """
        results = results.transpose(1, 0)[0]
        targets = targets.transpose(1, 0)[0]
        return self.criterion(results, targets)
    
    def accuracy(self, results, targets, condition):
        """ReEncoderの出力精度を計算する関数

        Args:
            results (torch.Tensor): ReEncoderの出力値
            targets (torch.Tensor): グラフ特徴量のラベル
            condition (str): 精度を計算するグラフ特徴量の名前

        Returns:
            正解率(0 ~ 1.0)
        """
        results = results.transpose(1, 0)[0]
        targets = targets.transpose(1, 0)[0]
        correct_cnt = 0
        for pred, correct in zip(results, targets):
            # グラフ特徴量毎にラベルを基準にした正解と判定される範囲が事前定義されているので、その範囲内にあれば正確に予測されているとみなす
            if correct - self.acc_range[condition][0] <= pred and pred <= correct + self.acc_range[condition][1]:
                correct_cnt += 1
        return correct_cnt / len(results)
    

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import Parameters
    params = Parameters()
    model = ReEncoder(input_size=172, params=params, device="cpu")
    in_ = torch.randn([32,20,172])
    out_ = model(in_)
    # results = torch.Tensor([[1], [1.5], [2.0], [2.5], [3.0]])
    # targets = torch.Tensor([[2], [2], [2], [2], [2]])
    # model.accuracy(results, targets, params.condition_params[0])