"""
生のデータセットを前処理し、modelの入力形式に変換するモジュール.
"""

import joblib
import numpy as np
import torch

from graph_process import complex_networks, convert_to_dfs_code
import utils


def preprocess(params, train_directory='./dataset/train/', valid_directory='./dataset/valid/'):
    """
    生データセットからグラフオブジェクトを作成し、それとそのサイズを保存する関数.

        Args:
            params (config.Parameters)  : グローバル変数のセット  
            train_directory      (str)  : 前処理後のtrain用データセットが保存されるディレクトリ
                default: './dataset/train/'
            valid_directory      (str)  : 前処理後のvalidation用データセットが保存されるディレクトリ
                default: './dataset/valid/'

    """
    
    complex_network = complex_networks.ComplexNetworks()
    complex_network.make_twitter_graph_with_label()
    
    train_dfs,train_time_set,train_node_set,train_max_length,train_label = to_dfs_conditional(params.train_network_detail, params.dfs_mode)
    valid_dfs,valid_time_set,valid_node_set,valid_max_length,valid_label = to_dfs_conditional(params.valid_network_detail, params.dfs_mode)

    time_stamp_set = train_time_set | valid_time_set
    node_label_set = train_node_set | valid_node_set
    max_sequence_length = max(train_max_length, valid_max_length)
    conditional_label_length = params.condition_size #指定しているパラメータの数
    
    print(f"max_sequence_length = {max_sequence_length}")
    
    joblib.dump([len(time_stamp_set)+1, len(node_label_set)+1, 2, conditional_label_length], "dataset/param")

    time_dict = {time:index for index, time in enumerate(time_stamp_set)}
    node_dict = {node:index for index, node in enumerate(node_label_set)}

    del time_stamp_set, node_label_set
    
    get_onehot_and_list(train_dfs,time_dict,node_dict,max_sequence_length,train_label,train_directory, params.ignore_label)
    get_onehot_and_list(valid_dfs,time_dict,node_dict,max_sequence_length,valid_label,valid_directory, params.ignore_label)

def to_dfs_conditional(detail, dfs_mode):
    """
    labelが添付されたDFSコードを生成する関数
        
        Args:
            detail (dict)  : データセット作成に関する詳細
            dfs_mode (str) : DFSする前に、ノードidのlistをどのようにsortするか.

        Returns:
            () : dfs code
            () : time stamp set
            () : node label set
            () : max sequence length
            () : label sets
    """
    complex_network = complex_networks.ComplexNetworks()
    datasets, labelsets= complex_network.create_seq_conditional_dataset(detail)

    dfs_code = list()
    time_stamp_set = set()
    nodes_label_set = set()
    max_sequence_length = 0


    for graph in datasets:
        #covert_graph = graph_process.ConvertToDfsCode(graph)
        covert_graph = convert_to_dfs_code.ConvertToDfsCode(graph, mode=dfs_mode)
        tmp = covert_graph.get_dfs_code()
        # 一旦tmpにdfscodeを出してからdfscodeにappend
        dfs_code.append(tmp)
        # グラフの中の最大のシーケンス長を求める　+1はeosが最後に入る分
        if max_sequence_length < len(tmp)+1:
            max_sequence_length = len(tmp)+1

        time_u = set(tmp[:, 0])
        time_v = set(tmp[:, 1])
        time = time_u | time_v
        time_stamp_set = time_stamp_set| time

        node_u = set(tmp[:,2])
        node_v = set(tmp[:,3])
        node = node_u | node_v
        nodes_label_set = nodes_label_set | node

    return dfs_code, time_stamp_set, nodes_label_set,\
        max_sequence_length, labelsets

def get_onehot_and_list(dfs_code, time_dict, node_dict, max_sequence_length, label_set, directory, ignore_label):
    """指定されたサイズのone-hotへ変換する関数

    Args:
        dfs_code (_type_): _description_
        time_dict (_type_): _description_
        node_dict (_type_): _description_
        max_sequence_length (_type_): _description_
        label_set (_type_): _description_
        directory (_type_): _description_
        ignore_label (int): 5-tuples内で無視するラベル
    """
    time_end_num = len(time_dict.keys())
    node_end_num = len(node_dict.keys())
    dfs_code_onehot_list = []
    t_u_list = []
    t_v_list = []
    n_u_list = []
    n_v_list = []
    e_list = []
    for data in dfs_code:
        data = data.T
        # IDに振りなおす
        t_u = [time_dict[t] for t in data[0]]
        t_u.append(time_end_num)
        t_u = np.array(t_u)
        t_u_list.append(t_u)
        t_v = [time_dict[t] for t in data[1]]
        t_v.append(time_end_num)
        t_v = np.array(t_v)
        t_v_list.append(t_v)
        n_u = [node_dict[n] for n in data[2]]
        n_u.append(node_end_num)
        n_u = np.array(n_u)
        n_u_list.append(n_u)
        n_v = [node_dict[n] for n in data[3]]
        n_v.append(node_end_num)
        n_v = np.array(n_v)
        n_v_list.append(n_v)
        e = data[4]
        e = np.append(e,1)
        e_list.append(e)

        onehot_t_u = utils.convert2onehot(t_u,time_end_num+1)
        onehot_t_v = utils.convert2onehot(t_v,time_end_num+1)
        onehot_n_u = utils.convert2onehot(n_u,node_end_num+1)
        onehot_n_v = utils.convert2onehot(n_v,node_end_num+1)
        onehot_e = utils.convert2onehot(e,1+1)

        dfs_code_onehot_list.append(\
            np.concatenate([onehot_t_u,onehot_t_v,onehot_n_u,onehot_n_v,onehot_e],1))
    
    dfs_code_onehot_list = torch.Tensor(utils.padding(dfs_code_onehot_list,max_sequence_length,0))
    t_u_list = torch.LongTensor(utils.padding(t_u_list,max_sequence_length,ignore_label))
    t_v_list = torch.LongTensor(utils.padding(t_v_list,max_sequence_length,ignore_label))
    n_u_list = torch.LongTensor(utils.padding(n_u_list,max_sequence_length,ignore_label))
    n_v_list = torch.LongTensor(utils.padding(n_v_list,max_sequence_length,ignore_label))
    e_list = torch.LongTensor(utils.padding(e_list,max_sequence_length,ignore_label))
    
    joblib.dump(dfs_code_onehot_list,directory+'onehot')
    joblib.dump([t_u_list,t_v_list,n_u_list,n_v_list,e_list],directory+'label')
    joblib.dump(label_set,directory+'conditional')


if __name__ == "__main__":
    preprocess(train_network_detail, valid_network_detail)