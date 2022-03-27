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
  - こちらのURL(https://pytorch.org/get-started/previous-versions/)を参照のこと.

## Usage
- メインプログラムを実行．
  - `result/[日付][実行時刻]/` 下に実行結果とログが出力されます．
```shell
python main.py
```
- ReEncoderの事前学習
```
python train.py
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
- 指定できるパラメータは以下の通り.
```json
```

## Directory Structure
- プロジェクトの構成
```shell
```