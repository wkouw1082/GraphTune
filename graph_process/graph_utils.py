"""
よく使用されるグラフに関する処理をまとめたモジュール.

主な機能は以下.
・グラフオブジェクトから隣接行列への変換
・テキストファイルからグラフオブジェクトを作成
・グラフオブジェクトのグラフ特徴量を計算し, csvへ出力
"""

import networkx as nx
import numpy as np


def graph_obj2mat(G):
    """networkxのグラフオブジェクトを隣接行列に変換する関数
    
    同じラベルを持つノードはない前提.

    Args:
        G (networkx.classes.graph.Graph): networkxのグラフオブジェクト

    Returns:
        (numpy.ndarray) : 隣接行列
    """
    nodes = G.nodes
    edges = G.edges
    nodes = {i: node_label for i, node_label in enumerate(nodes)}

    adj_mat = np.zeros((len(nodes), len(nodes)))

    # forでぶん回している. smartにしたい
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]

        node1_arg = None
        node2_arg = None
        for key, node_label in nodes.items():
            if node1 == node_label:
                node1_arg = key
            if node2 == node_label:
                node2_arg = key

            # for短縮のため
            if not node1_arg is None and not node2_arg is None:
                break
        adj_mat[node1_arg, node2_arg] = 1
        adj_mat[node2_arg, node1_arg] = 1
    return adj_mat

def text2graph(text_files):
    """
    テキストファイルからグラフオブジェクトを作成する関数
    
        Args:
            text_files (list) : text fileのリスト
        
        Returns:
            (list) : グラフオブジェクトのリスト
    """
    graph_data = []
    for text_file in text_files:
        with open(text_file, 'rb') as f:
            G = nx.read_edgelist(f, nodetype=int)
        graph_data.append(G)
    return graph_data



def dfs_code_to_graph_obj(dfs_code,end_value_list, edge_num=None):
    """DFScodeをnetworkxのグラフオブジェクトに変換する関数

    Args:
        dfs_code ([np.array]): [(sequence,5)のnp.array]
        end_value_list ([list]): [終了コード[5]]

    Returns:
        [networkx_graph]: [networkxのグラフオブジェクトを返す]
    """    
    G = nx.Graph()
    for current_code in dfs_code:
        for i in range(len(current_code)):
            if edge_num:
                if len(G.edges) >= edge_num:
                    return G
                elif current_code[i] == end_value_list[i]-1:
                    return G
            else:
                # 長さ自体はend_value_listの値だが実際の値は0から始まっているため-1する
                if current_code[i] == end_value_list[i]-1:
                    return G

        tu,tv,_,_,_ = current_code
        G.add_edge(tu,tv)
    return G


if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1 ,3), (3, 4), (4, 5)])
    mat = graph_obj2mat(G)
    print(type(mat))
    print(mat)