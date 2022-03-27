"""
グラフに関する処理をまとめたモジュール.

主な機能は以下の通り.
・生のデータセットを読み込み、グラフオブジェクトに変換して保存する
・グラフオブジェクトからグラフ特徴量を算出し、csvファイルに書き出す
"""

import glob
import networkx as nx
from sklearn.model_selection import train_test_split
import joblib
import torch
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graph_process import graph_statistic
from graph_process.graph_utils import text2graph
from config import Parameters


class ComplexNetworks:
    """
    複雑ネットワークモデルを使用してデータセットを作成するクラス.
    """

    def __init__(self):
        params = Parameters()
        self.test_size = params.test_size
        self.condition_params = params.condition_params
        self.condition_round = params.condition_round
        self.twitter_train_path = params.twitter_train_path
        self.twitter_valid_path = params.twitter_valid_path
        self.twitter_path = params.twitter_path

    def create_seq_conditional_dataset(self, detail):
        """
        条件付きシーケンスデータセットを作成する関数.

            Args:
                detail (dict) : データセット作成に関する詳細
                
            Returns:
                (nx.graph) : グラフオブジェクト形式のデータセット
                (torch.tensor) : condition情報をもとに計算されたlabel
        """
        for i, (key, value) in enumerate(detail.items()):
            generate_num = value[0]
            data_dim = value[1]
            params = value[2]
            for param in params:
                if key=='twitter_train':
                    datasets = joblib.load(self.twitter_train_path)
                    dataset, label = datasets[0], datasets[1]
                elif key=='twitter_valid':
                    datasets = joblib.load(self.twitter_valid_path)
                    dataset, label = datasets[0], datasets[1]
        return dataset, label

    def make_twitter_graph_with_label(self):
        """
        テキストファイルからグラフオブジェクトを作成し、trainとvalidに分割して保存する関数.
        また、グラフ統計量を計算し、labelとして保存する.
        """
        text_files = glob.glob(self.twitter_path)
        graph_data = text2graph(text_files)
        train_data, valid_data = train_test_split(graph_data, test_size=self.test_size, random_state=0, shuffle=True)

        train_labels = torch.Tensor()
        valid_labels = torch.Tensor()

        st = graph_statistic.GraphStatistic()
        
        # conditionalで指定するlabelを取得する
        print('generate train data ...')
        for graph in tqdm(train_data):
            # クラスタ係数と最長距離を指定するためにパラメータを取得してlabelとする
            # paramsはリスト型で渡されるのでindex[0]をつける
            params = st.calc_graph_traits2csv([graph],self.condition_params)[0]
            tmp_label = []
            for param in params.values():
                tmp_label.append(round(param, self.condition_round))
            if len(tmp_label) == 1:
                tmp_label = torch.tensor(tmp_label).float().unsqueeze(0)
            else:
                tmp_label = torch.tensor(np.array([np.prod(tmp_label)])).float().unsqueeze(0)
            train_labels = torch.cat((train_labels, tmp_label),dim=0)
        train_labels.unsqueeze(1)

        print('generate valid data ...')
        for graph in tqdm(valid_data):
            # クラスタ係数と最長距離を指定するためにパラメータを取得してlabelとする
            # paramsはリスト型で渡されるのでindex[0]をつける
            params = st.calc_graph_traits2csv([graph], self.condition_params)[0]
            tmp_label = []
            for param in params.values():
                tmp_label.append(round(param, self.condition_round))
            if len(tmp_label) == 1:
                tmp_label = torch.tensor(tmp_label).float().unsqueeze(0)
            else:
                tmp_label = torch.tensor(np.array([np.prod(tmp_label)])).float().unsqueeze(0)
            valid_labels = torch.cat((valid_labels, tmp_label),dim=0)
        valid_labels.unsqueeze(1)

        joblib.dump([train_data, train_labels], self.twitter_train_path)
        joblib.dump([valid_data, valid_labels], self.twitter_valid_path)
    




if __name__ == "__main__":
    from config import Parameters
    params = Parameters()

    complex_networks = ComplexNetworks()
    complex_networks.create_seq_conditional_dataset(params.train_generate_detail)