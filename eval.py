"""
学習済みモデルを使用して, 条件付き生成をするモジュール.
"""

import os
import argparse
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import shutil
import joblib
import networkx as nx

from config import common_args, Parameters
import utils
from utils import dump_params, setup_params
from utils import set_logging, make_dir, get_gpu_info
import preprocess as pp
from models import cvae
from graph_process import graph_utils


def eval(params, args):
    """学習済みモデルを使用して, グラフを生成する関数

	Args:
		params (config.Parameters)  : global変数のset
		args   (argparse.Namespace) : コンソールの引数のset
	"""
	# Device
    device = utils.get_gpu_info()

    # Load preprocessed dataset
    train_label = joblib.load("dataset/train/label")
    time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
    dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
    print("--------------")
    print(f"time size: {time_size}")
    print(f"node size: {node_size}")
    print(f"edge size: {edge_size}")
    print(f"conditional size: {conditional_size}")
    print("--------------")

    # 生成されたグラフが十分なサイズであるか判別する関数
    is_sufficient_size = lambda graph: True if graph.number_of_nodes() > params.size_th else False

    # model選択
    model = cvae.CVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device).to(device)
    if args.eval_model:
        model.load_state_dict(torch.load(args.eval_model, map_location="cpu"))
        model.eval()
        eval_dir = "result/" + args.eval_model.split("/")[1] + "/eval/"
    else:
        print("評価するモデルが選択されていません!")
        exit()

    # 条件付き生成用のconditionを作成
    # conditional_label = [[params.condition_values["Average path length"][i]] for i in range(3)]
    # conditional_label = torch.tensor(conditional_label)
    conditional_labels = utils.get_condition_values(params.condition_params, params.condition_values)

    # 条件付き生成
    result_low = model.generate(300, torch.tensor([conditional_labels[0]]).float(), max_size=params.generate_edge_num)
    result_middle = model.generate(300, torch.tensor([conditional_labels[1]]).float(),
                                   max_size=params.generate_edge_num)
    result_high = model.generate(300, torch.tensor([conditional_labels[2]]).float(), max_size=params.generate_edge_num)

    result_all = [result_low, result_middle, result_high]

    results = {}
    generated_keys = []
    dfs_codes = torch.Tensor().cpu()

    for index, (result, cond_label) in enumerate(zip(result_all, conditional_labels)):
        # generated graphs
        result = [code.unsqueeze(2) for code in result]
        dfs_code = torch.cat(result, dim=2)
        # dfs_codes = torch.cat((dfs_codes,dfs_code.cpu()))
        generated_graph = []
        for code in dfs_code:
            graph = graph_utils.dfs_code_to_graph_obj(
                code.cpu().detach().numpy(),
                [time_size, time_size, node_size, node_size, edge_size],
                edge_num=params.generate_edge_num)
            if nx.is_connected(graph) and is_sufficient_size(graph):
                generated_graph.append(graph)

        joblib.dump(generated_graph, eval_dir + "_".join(params.condition_params) + "_" + str(cond_label) + '.pkl')
    print("eval is completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = common_args(parser)
    parser.add_argument('--eval_model')  # 学習済みモデルのパス
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得

    # eval
    eval(params, args)
