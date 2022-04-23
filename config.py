"""
プロジェクト内のパラメータを管理するためのモジュール．

A) プログラムを書くときにやること．
  1) デフォルトパラメータを `Parameters` クラス内で定義する．
  2) コマンドライン引数を `common_args` 内で定義する．

B) パラメータを指定して実行するときにやること．
  1) `python config.py` とすると，デフォルトパラメータが `parameters.json` というファイルに書き出される．
  2) パラメータを指定する際は，Parametersクラスを書き換えるのではなく，jsonファイル内の値を書き換えて，
  `python -p parameters.json main.py`
  のようにjsonファイルを指定する．
"""

from dataclasses import dataclass, field
from utils import dump_params
from argparse import ArgumentParser


@dataclass(frozen=True)
class Parameters:
    """
    プログラム全体を通して共通のパラメータを保持するクラス．
    ここにプロジェクト内で使うパラメータを一括管理する．
    """
    args: dict = field(default_factory=lambda: {})  # コマンドライン引数
    run_date: str = ''  # 実行時の時刻
    git_revision: str = ''  # 実行時のプログラムのGitのバージョン
    
    # Condition
    conditional_mode: bool = True   # 条件付き学習, 条件付き生成を行う場合はTrue
    condition_params: list = field(default_factory=lambda: ["Average path length"])  # preprocessでconditionとして与えるparameter
    condition_round: int = 4    # conditionの値の丸める桁数
    condition_size: int = 1     # condition size
    condition_values: dict = field(default_factory=lambda: {
        "Power-law exponent": [2.6, 3.0, 3.4], 
        "Clustering coefficient":[0.1,0.2,0.3], 
        "Average path length":[3.0,4.0,5.0], 
        "Average degree":[3,4,5], 
        "Edge density":[0.05,0.075,0.10], 
        "Modularity":[0.5, 0.6, 0.7], 
        "Diameter":[10,20,30], 
        "Largest component size":[7.0, 14.0, 20.0],
    })

    # Preprocess
    split_size: dict = field(default_factory=lambda: {"train": 0.8, "valid": 0.1, "test": 0.1}) # データセットをtrain用とvalid用に分割する際のvalid用データの比率
    reddit_path: str = "./data/reddit_threads/reddit_edges.json"
    twitter_path: str = "./data/edgelists_50/renum*"
    twitter_train_path: str = './data/twitter_train'
    twitter_valid_path: str = './data/twitter_valid'
    twitter_test_path: str  = './data/twitter_test'
    train_network_detail: dict = field(default_factory=lambda: {"twitter_train":[None,None,[None]]})  # preprocessでのtrain datasetの詳細. valueは[生成数, データ次元, [データセットの名前]].
    valid_network_detail: dict = field(default_factory=lambda: {"twitter_valid":[None,None,[None]]})  # preprocessでのvalid datasetの詳細. valueは[生成数, データ次元, [データセットの名前]].
    test_network_detail: dict  = field(default_factory=lambda: {"twitter_test" :[None,None,[None]]})  # preprocessでのtest datasetの詳細.  valueは[生成数, データ次元, [データセットの名前]].
    dfs_mode: str = "normal"  # ["high_degree_first", "normal", "low_degree_first"]から選択できる.
    ignore_label: int = 1500  # 5-tuples内のデータにおいて、無視するデータ
    
    # Graph properties
    power_degree_border_line: float = 0.7 # 次数分布の冪指数を出すときに大多数のデータに引っ張られるせいで１次元プロットが正しい値から離れてしまうのでいくつかの値を除いて導出するための除く割合
    
    
    # Models
    dropout: float = 0.5        # dropout層に入力されたデータをdropさせる割合
    word_drop_rate: float = 0   # word drop rate
    ## model hyper parameters
    model_params: dict = field(default_factory=lambda: {'batch_size': 37, 'clip_th': 10, 'de_hidden_size': 250, 'emb_size': 227, "re_en_hidden_size": 223, 'en_hidden_size': 223,
                                                       'lr': 0.001, 'rep_size': 10, "re_en_rep_size": 1, 'weight_decay': 0,
                                                       "alpha": 1, "beta": 3, "gamma": 3})
    """model parameters
    以前、tuningして得たbest params
    'lr' : 0.0015181790179257975
    'weight_decay' : 0.005663866734065268
    'clip_th' : 0.002362780918168105
    """
    
    # Train
    epochs: int = 30000   # エポック数
    model_save_point: int = 500  # modelをsaveするチェックポイント(エポック数)
    
    # Eval
    # 現状、"power_degree", "cluster_coefficient", "distance", "size"
    eval_params: list = field(default_factory=lambda: ["Power-law exponent", "Clustering coefficient", "Average path length", "Average degree" ,"Edge density", "Modularity", "Diameter","Largest component size"])
    sampling_generation: bool = True    # 生成時に出力される分布からサンプリングするか最大値から取るか
    generate_edge_num: int = 100        # 生成するgraphのエッジの数
    size_th: int = 0                    # 評価に用いるネットワークのサイズの閾値

    # Visualize
    visualize_detail: dict = field(default_factory=lambda: {
      "twitter_pickup": [300, None, [None]]  # 統計的手法の場合, [生成するグラフ数, データ次元数, [指定パラメータ1, 指定パラメータ2]]
    })
    visualize_types: dict = field(default_factory=lambda: {"Real_data":'bbb',\
        "AveragePathLength_3.0":'aaa',"AveragePathLength_0.4":'yyy',"Average_PathLength_0.5":'xxx'})


def common_args(parser: 'ArgumentParser'):
    """
    コマンドライン引数を定義する関数．

        Args:
            parser (:obj: ArgumentParser):
    """
    parser.add_argument("-p", "--parameters", help="パラメータ設定ファイルのパスを指定．デフォルトはNone", type=str, default=None)
    parser.add_argument("-a", "--arg1", type=int, help="arg1の説明", default=0)  # コマンドライン引数を指定
    parser.add_argument("--arg2", type=float, help="arg2の説明", default=1.0)  # コマンドライン引数を指定
    return parser


if __name__ == "__main__":
    dump_params(Parameters(), './', partial=True)  # デフォルトパラメータを
