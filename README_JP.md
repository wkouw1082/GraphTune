# GraphTune: A Learning-Based Graph Generative Model With Tunable Structural Features

このプロジェクトは，長岡技術科学大学 通信ネットワーク研究室において開発されたグラフ生成モデル GraphTune の PyTorch による公式実装です．  
GraphTuneのモデルの説明と評価は，[我々の論文](https://arxiv.org/abs/2201.11494)を参照してください．

あなたのプロジェクトに本コードが役立ちましたら，以下の論文を引用をお願いします．

> Kohei Watabe, Shohei Nakazawa, Yoshiki Sato, Sho Tsugawa, and Kenji Nakagawa, ``GraphTune: A Learning-Based Graph Generative Model With Tunable Structural Features'', IEEE Transactions on Network Science and Engineering, 2023.

```bibtex
@ARTICLE{GraphTune,
  author={Watabe, Kohei and Nakazawa, Shohei and Sato, Yoshiki and Tsugawa, Sho and Nakagawa, Kenji},
  journal={IEEE Transactions on Network Science and Engineering}, 
  title={GraphTune: A Learning-Based Graph Generative Model With Tunable Structural Features}, 
  year={2023},
  volume={},
  number={},
  doi={10.1109/TNSE.2023.3244590}
}
```


## Requirement
- Python 3.7.6
- pytorch 1.9.0

## Installation
- 前処理後のデータセットが保存されるディレクトリを作成
```shell
mkdir dataset
```
- 結果出力用ディレクトリを作成
```shell
mkdir result
```
- 仮想環境の構築
```shell
python -m venv venv
```
- 各種モジュールのインストール
```shell
pip install --upgrade pip
pip install -r requirements.txt
```
- pytorchのインストール
  - 以下のURLを参照のこと.
    - https://pytorch.org/get-started/previous-versions/

## Training
- GraphTuneの学習をバックグラウンドで実行(`train.py`を修正する必要あり).
```shell
nohup python -u train.py --preprocess --use_model cvae &
```
- Modelのチェックポイントから学習を再開する例(前処理されたdatasetは統一することに注意).
```shell
nohup python -u train.py --use_model cvae --checkpoint_file result/{結果出力ファイル名}/train/weight_35000 --init_epoch 35001 &
```
- ReEncoderの事前学習
```shell
nohup python -u train.py --preprocess --use_model re_encoder &
```
- 事前学習済みReEncoderを用いたCVAEの学習の例.
```shell
nohup python -u train.py --preprocess --use_model cvae_with_re_encoder --re_encoder_file result/{結果出力ファイル名}/train/valid_best_weight &
```
- 2-tuplesのDFSコードを扱うGraphTuneの学習
```shell
nohup python -u train.py --preprocess --preprocess_type dfs_2_tuples --use_model cvae_for_2_tuples &
```

## Evaluation
- 学習済みモデルを使用して条件付きグラフ生成をする例.
```shell
python eval.py --use_model cvae --eval_model result/{結果出力ファイル名}/train/valid_best_weight
```

## Visualization
- 生成されたグラフをplotする例.
```shell
python visualize.py --eval_graphs result/{結果出力ファイル名}/eval/
```
- Ubuntuの場合は, 計算グラフを`Graphviz`を使用して, 以下の手順で可視化できる.
  - `Graphviz`ライブラリのinstall
```shell
apt install -y --no-install-recommends graphviz graphviz-dev
```
- 
  - 計算グラフの可視化をするためのプログラムの例を以下に示す.
  - 結果はrootディレクトリの中に, デフォルトではpdf形式で保存される.
```python
from torchviz import make_dot
x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)
y = x1 * x2
dot = make_dot(y)
dot.render("example_file_name")
```
- 
  - modelの出力から計算グラフを可視化する場合.
```python
from torchviz import make_dot
y = model(x)
# paramsを指定することにより, 計算グラフにmodelクラス内で定義された変数名が対応する箇所(node)に記載される.
image = make_dot(y, params=dict(model.named_parameters()))
image.format("png") # png形式で保存する
image.render("example_model_file_name")
```


## Parameter Settings
- 指定できるパラメータは以下の通り. 詳細は`config.py`を参照.
```json
{
    "args": {
        "parameters": null,
        "arg1": 0,
        "arg2": 1.0,
        "preprocess": true,
        "use_model": "VAEwithReEncoder",
        "re_encoder_file": null,
        "checkpoint_file": null,
        "init_epoch": null
    },
    "run_date": "20220505_212813",
    "git_revision": "d8bcffadc7c4de909d2407c4e6297df924566908\n",
    "conditional_mode": true,
    "condition_params": [
        "Average path length"
    ],
    "condition_round": 4,
    "condition_size": 1,
    "condition_values": {
        "Power-law exponent": [
            2.6,
            3.0,
            3.4
        ],
        "Clustering coefficient": [
            0.1,
            0.2,
            0.3
        ],
        "Average path length": [
            3.0,
            4.0,
            5.0
        ],
        "Average degree": [
            3,
            4,
            5
        ],
        "Edge density": [
            0.05,
            0.075,
            0.1
        ],
        "Modularity": [
            0.5,
            0.6,
            0.7
        ],
        "Diameter": [
            10,
            20,
            30
        ],
        "Largest component size": [
            7.0,
            14.0,
            20.0
        ]
    },
    "split_size": {
        "train": 0.9,
        "valid": 0.1,
        "test": 0
    },
    "reddit_path": "./data/reddit_threads/reddit_edges.json",
    "twitter_path": "./data/edgelists_50/renum*",
    "twitter_train_path": "./data/twitter_train",
    "twitter_valid_path": "./data/twitter_valid",
    "twitter_test_path": "./data/twitter_test",
    "train_network_detail": {
        "twitter_train": [
            null,
            null,
            [
                null
            ]
        ]
    },
    "valid_network_detail": {
        "twitter_valid": [
            null,
            null,
            [
                null
            ]
        ]
    },
    "test_network_detail": {
        "twitter_test": [
            null,
            null,
            [
                null
            ]
        ]
    },
    "dfs_mode": "normal",
    "ignore_label": 1500,
    "power_degree_border_line": 0.7,
    "dropout": 0.5,
    "word_drop_rate": 0,
    "model_params": {
        "batch_size": 37,
        "clip_th": 10,
        "de_hidden_size": 250,
        "emb_size": 227,
        "re_en_hidden_size": 223,
        "en_hidden_size": 223,
        "lr": 0.001,
        "rep_size": 10,
        "re_en_rep_size": 1,
        "weight_decay": 0,
        "alpha": 1,
        "beta": 3,
        "gamma": 3
    },
    "epochs": 100000,
    "model_save_point": 500,
    "eval_params": [
        "Power-law exponent",
        "Clustering coefficient",
        "Average path length",
        "Average degree",
        "Edge density",
        "Modularity",
        "Diameter",
        "Largest component size"
    ],
    "sampling_generation": true,
    "generate_edge_num": 100,
    "size_th": 0,
    "visualize_detail": {
        "twitter_pickup": [
            300,
            null,
            [
                null
            ]
        ]
    },
    "visualize_types": {
        "Real_data": "bbb",
        "AveragePathLength_3.0": "aaa",
        "AveragePathLength_0.4": "yyy",
        "Average_PathLength_0.5": "xxx"
    },
    "acc_range": {
        "Power-law exponent": [
            0.1,
            0.1
        ],
        "Clustering coefficient": [
            0.01,
            0.01
        ],
        "Average path length": [
            0.05,
            0.05
        ],
        "Average degree": [
            0.1,
            0.1
        ],
        "Edge density": [
            0.005
        ],
        "Modularity": [
            0.02,
            0.02
        ],
        "Diameter": [
            0,
            0
        ],
        "Largest component size": [
            0,
            0
        ]
    }
}
```

## Directory Structure
- プロジェクトの構成
```shell
.
├── data                    # dataset
│   ├── Twitter
│   │   └── edgelists
│   ├── Twitter_2000
│   ├── csv
│   ├── edgelists_50
│   └── reddit_threads
├── dataset                 # 前処理されたdataset
│   ├── train
│   └── valid
├── graph_process           # グラフ処理に関するモジュール
│  
├── models                  # 機械学習モデル
```

## 新規の機械学習モデルの開発と学習
- 新しく機械学習モデルを作成する場合は、`models`ディレクトリ直下にpythonファイルで作成する
- 新規作成したモデルの識別名を考え、`config.py`のParameters()のmodel_setにその識別名を追加する
- `train.py`に新規モデルの学習に必要な処理を追加する


## 連絡先
本プロジェクトに関する問い合わせは，以下までご連絡ください．

長岡技術科学大学 大学院工学研究科  
准教授 渡部 康平  
住所: 〒940-2188 新潟県 長岡市 上富岡町 1603-1  
Tel: +81-258-47-9537  
E-mail: [k_watabe@vos.nagaokaut.ac.jp](k_watabe@vos.nagaokaut.ac.jp)  
Web: [NUT ComNets Lab.](https://kaede.nagaokaut.ac.jp/)
