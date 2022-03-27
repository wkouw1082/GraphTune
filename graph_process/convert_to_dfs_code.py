"""
networkx形式のグラフオブジェクトからDFSコードへ変換するモジュール.
"""

import networkx as nx
from collections import OrderedDict
import numpy as np


class ConvertToDfsCode():
    """
    グラフをDFSコードへ変換するクラス.
    """
    
    def __init__(self, graph, mode="normal"):
        """
        Args:
            graph (nx.graph) : networkx形式のグラフオブジェクト
            mode (str) : DFSする順番
                You can select from ["normal", "low_degree_first", "high_degree_first"].
                default: "normal"
        """
        self.G = graph
        self.node_tree = [node for node in graph.nodes()]
        self.edge_tree = [edge for edge in graph.edges()]
        self.dfs_code = list()
        self.visited_edges = list()
        self.time_stamp = 0
        self.node_time_stamp = [-1 for i in range(graph.number_of_nodes())]
        self.mode=mode

    def get_max_degree_index(self):
        max_degree = 0
        max_degree_index = 0
        for i in range(self.G.number_of_nodes()):
            if(self.G.degree(i) >= max_degree):
                max_degree = self.G.degree(i)
                max_degree_index = i

        return max_degree_index

    def dfs(self,current_node):
        neightbor_node_dict = OrderedDict({neightbor:self.node_time_stamp[neightbor] for neightbor in self.G.neighbors(current_node)})
        neighbor_degree_dict = OrderedDict({neighbor: self.G.degree[neighbor] for neighbor in neightbor_node_dict.keys()})
        if self.mode=="high_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=True))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        elif self.mode=="low_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=False))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        else:
            sorted_neightbor_node = OrderedDict(sorted(neightbor_node_dict.items(), key=lambda x: x[1], reverse=True))

        if(len(self.visited_edges) == len(self.edge_tree)):
            return

        for next_node in sorted_neightbor_node.keys():
            # visited_edgesにすでに訪れたエッジの組み合わせがあったらスルー
            if((current_node, next_node) in self.visited_edges or (next_node, current_node)in self.visited_edges):
                continue
            else:
                if(self.node_time_stamp[next_node] != -1):
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1

                    self.visited_edges.append((current_node,next_node))
                    # print(f"{current_node} => {next_node}")
                    self.dfs_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),0])
                else:
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1
                    # 次のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[next_node] == -1):
                        self.node_time_stamp[next_node] = self.time_stamp
                        self.time_stamp += 1
                    # timeStamp_u, timeStamp_v, nodeLabel u, nodeLable_v ,edgeLable(u,v)の順のリストを作成
                    # print(f"{current_node} => {next_node}")
                    self.dfs_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),0])
                    self.visited_edges.append((current_node,next_node))
                    self.dfs(next_node)

    def get_dfs_code(self):
        self.dfs(self.get_max_degree_index())
        return np.array(self.dfs_code)