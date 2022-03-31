# GraphTune

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

## Usage
- メインプログラムを実行．
  - `result/[日付][実行時刻]/` 下に実行結果とログが出力されます．
```shell
python main.py
```
- Modelの学習をバックグラウンドで実行.
```shell
nohup python -u train.py &
```
- 学習済みモデルを使用して条件付きグラフ生成.
```shell
python eval.py --eval_model result/{結果出力ファイル名}/train/valid_best_weight
```
- 生成されたグラフをplotする.
```shell
python visualize.py --eval_graphs result/{結果出力ファイル名}/eval/
```
- デフォルトのパラメータ設定をjson出力．
```shell
python config.py  # parameters.jsonというファイルが出力される．
```
- 以下のように，上記で生成されるjsonファイルの数値を書き換えて，実行時のパラメータを指定できます．
```shell
python -p parameters.json main.py
```
- 詳しいコマンドの使い方は以下のように確認できます．
```shell
python main.py -h
```
- Ubuntuの場合は, 計算グラフを`Graphviz`を使用して, 以下の手順で可視化できます.
  - `Graphviz`ライブラリのinstall
```shell
apt install -y --no-install-recommends graphviz graphviz-dev
```
- 
  - 計算グラフの可視化をするためのプログラムの例を以下に示します.
  - 結果はrootディレクトリの中に, デフォルトではpdf形式で保存されます.
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
- 指定できるパラメータは以下の通り.詳細は`config.py`を参照.
```json
{
    "run_date": "20220331_015910",
    "git_revision": "c3c148f122670779ebebdbfc3bd46ae456c81fcb\n",
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
    "test_size": 0.1,
    "reddit_path": "./data/reddit_threads/reddit_edges.json",
    "twitter_path": "./data/edgelists_50/renum*",
    "twitter_train_path": "./data/twitter_train",
    "twitter_valid_path": "./data/twitter_eval",
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
    "dfs_mode": "normal",
    "ignore_label": 1500,
    "power_degree_border_line": 0.7,
    "dropout": 0.5,
    "word_drop_rate": 0,
    "model_params": {
        "batch_size": 37,
        "clip_th": 0.002362780918168105,
        "de_hidden_size": 250,
        "emb_size": 227,
        "re_en_hidden_size": 223,
        "en_hidden_size": 223,
        "lr": 0.0015181790179257975,
        "rep_size": 10,
        "re_en_rep_size": 1,
        "weight_decay": 0.005663866734065268,
        "alpha": 1,
        "beta": 3,
        "gamma": 3
    },
    "epochs": 20,
    "model_save_point": 10,
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
    "size_th": 0
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