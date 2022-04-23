"""
グラフ統計量(グラフ特徴量, グラフプロパティ)を算出するモジュール.

"""
from time import time_ns
from networkx.classes import graph
from networkx.readwrite import json_graph
import numpy as np
import networkx as nx
import torch
from collections import OrderedDict
import networkx.algorithms.approximation.treewidth as nx_tree
import networkx.algorithms.community as nx_comm
import community
from community import community_louvain
import random
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
import sympy as sym
from sympy.plotting import plot
import sys
import time
import math
import json
import glob
from tqdm import tqdm
import csv
import os
from sklearn.model_selection import train_test_split
import powerlaw
import pandas as pd
import argparse

from graph_process.graph_utils import graph_obj2mat
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import Parameters



class GraphStatistic:
    """
    グラフデータからグラフ統計量を算出するクラス.
    """

    def __init__(self):
        params = Parameters()
        self.power_degree_border_line = params.power_degree_border_line
    
    # 以下、グラフ統計量を計算する関数群
    def degree_dist(self, graph):
        """隣接行列を入力として次数分布を作成し,べき指数を計算する関数

            Args:
                graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

            Returns:
                (float) : 次数分布のべき指数
        
            Examples:
                >>> from config import Parameters
                >>> params = Parameters()
                >>> graph_statistic = GraphStatistic()
                >>> G = nx.Graph()
                >>> G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5)])
                >>> print(graph_statistic.degree_dist(G))
                -0.7324867603589632
        """
        degree = list(dict(nx.degree(graph)).values())

        import collections
        power_degree = dict(collections.Counter(degree))
        power_degree = sorted(power_degree.items(), key=lambda x:x[0])
        x = []
        y = []
        
        for i in power_degree:
            num = i[0]
            amount = i[1]
            x.append(num)
            y.append(amount)
        y = np.array(y) / sum(y)#次数を確率化
        sum_prob = 0
        for index,prob in enumerate(y):
            sum_prob += prob
            if sum_prob >= self.power_degree_border_line:
                border_index = index + 1
                break

        x_log = np.log(np.array(x))
        y_log = np.log(np.array(y))

        x_split_plot = x_log[border_index:]
        y_split_plot = y_log[border_index:]
        param =  np.polyfit(x_split_plot,y_split_plot,1)
        return param[0]
    
    def power_law_alpha(self, graph):
        """
        入力グラフの次数分布のpower law coefficientを計算する

            Args:
                graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

            Returns:
                (float) : power law coefficient
        """
        A_in = graph_obj2mat(graph)
        degrees = A_in.sum(axis=0).flatten()
        fit = powerlaw.Fit(degrees, discrete=True)
        return fit.power_law.alpha

    def cluster_coeff(self, graph):
        """平均クラスタ係数を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            float: 平均クラスタ係数
        """
        #graph = np.array(graph)
        #graph = mat2graph_obj(graph)
        return nx.average_clustering(graph)

    def ave_dist(self, graph):
        """平均最短経路長を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (float) : 平均最短経路長
        """
        #graph = np.array(graph)
        #graph = mat2graph_obj(graph)
        return nx.average_shortest_path_length(graph)

    def ave_degree(self, graph):
        """平均次数を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (float) : 平均次数
        """
        degree_count = 0
        degree_dict = graph.degree()
        for node in degree_dict:
            node_num, node_degree = node
            degree_count += node_degree
        return degree_count/graph.number_of_nodes()

    def density(self, graph):
        """グラフの密度を求める

        Args:
            graph (nx.graph): [計算したいnetworkx型のグラフ]

        Returns:
            [float]: [密度の値]
        """
        try:
            return nx.density(graph)
        except:
            return 0

    # 未完成
    def clique(self, graph):
        """グラフ内の最大クリークの数を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (int) : グラフ内の最大クリークの数
        """
        return nx.graph_number_of_cliques(graph)

    def modularity(self, graph):
        """グラフのmodularityを求める関数

        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ

        Returns:
            float: modularityの値
        """
        partition = community_louvain.best_partition(graph)
        return community_louvain.modularity(partition,graph)

    def number_of_clusters(self, graph):
        """クラスタの数を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (int) : クラスタの数
        """
        clusters_dict = community_louvain.best_partition(graph)
        clusters_set = set(value for key, value in clusters_dict.items())
        return len(clusters_set)

    def largest_component_size(self, graph):
        """最大コンポーネントサイズを計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (int) : 最大コンポーネントサイズ
        """
        largest_graph = max(nx.connected_components(graph), key=len)
        return len(largest_graph)

    # 未完成
    def giant_component(self, graph):
        """最大サイズのコンポーネントを計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (networkx.classes.graph.Graph) : 最大サイズのコンポーネント
        """
        print(graph.subgraph(max(nx.connected_components(graph), key=len)))
        return graph.subgraph(max(nx.connected_components(graph), key=len))

    def number_of_connected_component(self, graph):
        pass

    def degree_assortativity(self, graph):
        pass

    def reciprocity(self, graph):
        pass

    def maximum_of_shortest_path_lengths(self, graph):
        """最長の長さを持つ最短経路長の長さを求める関数

        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ

        Returns:
            int: 経路の長さ
        """
        max_shortest_path_length = 0
        path_dict = nx.shortest_path(graph)
        for node_num, paths in path_dict.items():
            for connect_node_num, path in paths.items():
                if len(path) >= max_shortest_path_length:
                    max_shortest_path_length = len(path)

        return max_shortest_path_length
        

    def degree_centrality(self, graph):
        '''
        グラフの各ノードの次数中心性を導出する
        
        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ
        
        Returns:
            (dict): グラフの各ノードの次数中心性のdict
        '''
        degree_centers = nx.degree_centrality(graph)
        return degree_centers

    def betweenness_centrality(self, graph):
        '''
        グラフの各ノードの媒介中心性を導出する
        
        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ
        
        Returns:
            (dict): グラフの各ノードの媒介中心性のdict
        '''
        betweenness_centers = nx.betweenness_centrality(graph)
        return betweenness_centers

    def closeness_centrality(self, graph):
        '''
        グラフの各ノードの近接中心性を導出する
        
        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ
        
        Returns:
            (dict): グラフの各ノードの近接中心性のdict
        '''
        closeness_centers = nx.closeness_centrality(graph)
        return closeness_centers
    
    def get_average_params(self, csv_path, save_dir):
        """生成したグラフの各パラメータごとの平均値を返すプログラム

        Args:
            csv_path (str): グラフごとのパラメータをもつcsvファイルのpath
            save_dir (str): 保存先のpath
        """
        file_name = os.path.splitext(os.path.basename(csv_path))[0]

        df = pd.read_csv(csv_path)
        average_df = df.mean()

        average_df.to_csv(save_dir + file_name + '.csv')
    
    @staticmethod
    def get_percentile_params(csv_path, save_dir, prob_list=(0.25, 0.5, 0.75)):
        """生成したグラフの各パラメータごとの中央値やパーセンタイルを返すプログラム

        Parameters
        ----------
        csv_path : str
            グラフごとのパラメータをもつcsvファイルのpath
        save_dir : str
            保存先のpath
        prob_list : list
            パーセンタイルを出力する確率の値
        """

        file_name = os.path.splitext(os.path.basename(csv_path))[0]

        df = pd.read_csv(csv_path)
        df_list = []
        for p in prob_list:
            df_list.append(df.quantile(p))
        percentile_df = pd.concat(df_list, axis=1)
        percentile_df.to_csv(f'{save_dir}{file_name}.csv')

    def calc_graph_traits2csv(self, graphs, eval_params):
        '''
        グラフごとにeval_paramsで指定されている特性値を計算してcsvへの保存形式に変換する関数
        
        Args:
            graphs: [graph_obj, ....]
            eval_params: 計算を行う特性値の名前のlist

        Returns:
            trait_list(list): 各グラフのparamのdictのlist
        '''
        trait_list=[]
        for index, graph in enumerate(graphs):
            tmp_dict = {}
            for key in eval_params:
                #if "id" in key:
                #    param = index
                if "Power-law exponent" in key:
                    try:
                        # param = self.degree_dist(graph)
                        param = self.power_law_alpha(graph)
                    except:
                        param = None
                if "Clustering coefficient" in key:
                    try:
                        param = self.cluster_coeff(graph)
                    except:
                        param = None
                if "Average path length" in key:
                    try:
                        param = self.ave_dist(graph)
                    except:
                        param = None
                if "Average degree" in key:
                    try:
                        param = self.ave_degree(graph)
                    except:
                        param = None
                if "Edge density" in key:
                    try:
                        param = self.density(graph)
                    except:
                        param = None
                if "Modularity" in key:
                    try:
                        param = self.modularity(graph)
                    except:
                        param = None
                if "Diameter" in key:
                    try:
                        param = self.maximum_of_shortest_path_lengths(graph)
                    except:
                        param = None
                if "degree_centrality" in key:
                    try:
                        param = self.degree_centrality(graph)
                    except:
                        param = None
                if "betweenness_centrality" in key:
                    try:
                        param = self.betweenness_centrality(graph)
                    except:
                        param = None
                if "closeness_centrality" in key:
                    try:
                        param = self.closeness_centrality(graph)
                    except:
                        param = None
                if "Largest component size" in key:
                    try:
                        param = self.largest_component_size(graph)
                    except:
                        param = None
                if "size" in key:
                    try:
                        param = graph.number_of_nodes()
                    except:
                        param = None
                tmp_dict.update({key:param})
            trait_list.append(tmp_dict)
        return trait_list

    def graph2csv(self, graphs, csv_dir, file_name, eval_params):
        '''各グラフデータのパラメータをcsvファイルに出力する関数

        Args:
            graphs      (list): グラフデータが格納されているリスト [GraphObj, ...]
            csv_dir   (string): csvファイルをの保存先ディレクトリ
                                e.g. csv_dir = "result/2021_0101/visualize/csv/"
            file_name (string): csvファイル名
                                e.g. file_name = "AveragePathLength_01"
            eval_params (list): 計算したいグラフ特徴量の名称のリスト
                                e.g. eval_params = ["Power-law exponent", "Clustering coefficient"]
        '''
        trait_dict = self.calc_graph_traits2csv(graphs, eval_params)
        with open(csv_dir + file_name + '.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=eval_params)
            writer.writeheader()
            writer.writerows(trait_dict)


if __name__ == "__main__":
    from config import Parameters
    params = Parameters()

    graph_statistic = GraphStatistic()
    G = nx.Graph()
    print(type(G))
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5)])
    trait_list = graph_statistic.calc_graph_traits2csv([G], params.eval_params)
    print(trait_list)