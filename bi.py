"""
可視化のバックエンド側の処理全般がまとめられたモジュール.
"""

import os
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

import utils


def scatter_diagram_visualize(eval_params, csv_path, output_path):
    """散布図を作成する関数
       なお、作成時にはeval paramsの全ての組み合わせが作成される

    Args:
        eval_params(list): グラフ特徴量のリスト
        csv_path    (str): 散布図を作成したいcsvfileのpath
        output_path (str): png形式の散布図を保存するディレクトリのpath
                           (例) output_path = "results/2021-01-01_00-00/visualize/scatter_diagram/"
    
    Examples:
        >>> scatter_diagram_visualize('./data/Twitter/twitter.csv')
    """
    dir_name = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    for param_v in eval_params:
        for param_u in eval_params:
            if re.search('centrality', param_v) or re.search('centrality', param_u) or param_v == param_u:
                continue
            fig = plt.figure()
            x_data = df[param_v]
            y_data = df[param_u]
            sns.jointplot(x=x_data,y=y_data,data=df)
            plt.savefig(output_path + dir_name + '/' + param_v + '_' + param_u + '.png')
            fig.clf()
            plt.close('all')

def histogram_visualize(eval_params, csv_path, output_path):
    """ヒストグラムを作成する関数

    Args:
        eval_params(list): グラフ特徴量のリスト
        csv_path    (str): ヒストグラムを作成したいcsvファイルのパス
        output_path (str): png形式のヒストグラムを保存するディレクトリのパス
                           (例) output_path = "results/2021-01-01_00-00/visualize/histogram/"
    """
    dir_name = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    for param in eval_params:
        fig = plt.figure()
        if re.search('centrality', param):
            # 全グラフのノードのパラメータを１つのリストにまとめる
            # 原因はわからないがなぜかstrで保存されてしまうのでdictに再変換:ast.literal_eval(graph_centrality)
            total_param = []
            for graph_centrality in df[param]:
                for centrality in ast.literal_eval(graph_centrality).values():
                    total_param.append(centrality)
            sns.histplot(total_param, kde=False)
        else:
            sns.kdeplot(df[param])
        plt.savefig(output_path + dir_name + '/' + param + '.png')
        plt.clf()
        plt.close('all')

def concat_scatter_diagram_visualize(eval_params, csv_paths, output_path):
    """散布図を結合する関数

    Args:
        eval_params(list): グラフ特徴量のリスト
        csv_paths    (str): ヒストグラムを作成したいcsvファイルのパス
        output_path (str): png形式のヒストグラムを保存するディレクトリのパス
                           (例) output_path = "results/2021-01-01_00-00/visualize/histogram/"
    """
    for param_v in eval_params:
        for param_u in eval_params:
            if re.search('centrality', param_v) or re.search('centrality', param_u) or param_v == param_u:
                continue
            fig = plt.figure()
            df = utils.concat_csv(csv_paths)
            sns.jointplot(x=df[param_v],y=df[param_u],data=df,hue='type')
            plt.savefig(output_path + param_v + '_' + param_u + '.png')
            fig.clf()
            plt.close('all')

def concat_histogram_visualize(eval_params, csv_paths, output_path):
    """複数のデータを結合したヒストグラムを作成する関数

    Args:
        eval_params (list): グラフ特徴量のリスト
        csv_paths   (list): ヒストグラムを作成するcsvファイルパスのリスト
        output_path (str) : png形式の結合ヒストグラムを保存するディレクトリのパス
                            (例) output_path = "results/2021-01-01_00-00/visualize/"
    """
    color_list = ['blue','red','green','gray']
    for param in eval_params:
        fig = plt.figure()
        for path, color in zip(csv_paths, color_list):
            df = pd.read_csv(path)
            # label_name = [key for key, value in visualize_types.items() if value == path][0]
            label_name = path.split("/")[-1]
            sns.kdeplot(df[param],label=label_name, color=color)

        plt.legend(frameon=True)
        plt.savefig(output_path + param + '.png')
        plt.clf()
        plt.close('all')

def pair_plot(eval_params, csv_paths, output_path):
    """Pair plotを作成する関数

    Args:
        eval_params(list): グラフ特徴量のリスト
        csv_paths    (str): ヒストグラムを作成したいcsvファイルのパス
        output_path (str): png形式のヒストグラムを保存するディレクトリのパス
                           (例) output_path = "results/2021-01-01_00-00/visualize/histogram/"
    """
    fig = plt.figure()
    df = utils.concat_csv(csv_paths)
    df = df.reindex(columns=eval_params + ['type'])
    markers = ["o", "s", "D", "X"][0:df['type'].nunique()]  # 必要なマーカー数だけ取り出す
    sns.pairplot(df,data=df,hue='type',markers=markers, plot_kws=dict(alpha=0.25))
    plt.savefig(output_path + 'pair_plot.pdf')
    plt.savefig(output_path + 'pair_plot.png', dpi=300)
    fig.clf()
    plt.close('all')

def graph_visualize(graphs, file_name_list, output_path, sampling_num=10):
    """グラフを可視化する関数

    Args:
        graphs          (list): グラフオブジェクトのリスト
        file_name_list   (str): 保存ファイル名
        output_path      (str): 出力先ディレクトリ
                                 e.g. output_path = "result/20220329_171445/visualize/graph_structure/"
    """
    for i, graph in enumerate(graphs):
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(graph)
        nx.draw_networkx(graph, pos)
        plt.axis("off")
        plt.savefig(output_path + file_name_list[i] + '.png')
        plt.clf()
        plt.close('all')