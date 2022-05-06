"""
学習済みモデルを使用して, 条件付き生成をするモジュール.
"""

import os
import logging
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
from models import cvae, cvae_for_2_tuples, cvae_with_re_encoder
from graph_process import graph_utils


def eval(params, logger):
    """CVAE用の学習済みモデルを使用して, グラフを生成する関数

	Args:
		params (config.Parameters)  : global変数のset
		args   (argparse.Namespace) : コンソールの引数のset
	"""
	# Deviceはcpuに限定
    device = "cpu"

    # Load preprocessed dataset
    time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
    logger.info("--------------")
    logger.info(f"time size: {time_size}")
    logger.info(f"node size: {node_size}")
    logger.info(f"edge size: {edge_size}")
    logger.info(f"conditional size: {conditional_size}")
    logger.info("--------------")

    # 生成されたグラフが十分なサイズであるか判別する関数
    is_sufficient_size = lambda graph: True if graph.number_of_nodes() > params.size_th else False

    # model選択
    use_model = params.args['use_model']
    if use_model == 'cvae':
        dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
        model = cvae.CVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
    elif use_model == 'cvae_for_2_tuples':
        dfs_size = 2 * time_size + conditional_size
        model = cvae_for_2_tuples.CVAE(dfs_size, time_size, conditional_size, params, device)
    elif use_model == 'cvae_with_re_encoder':
        dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
        model = cvae_with_re_encoder.CVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
    else:
        print("評価するモデルが選択されていません!")
        exit()
    
    # 評価対象のモデルをload
    if params.args['eval_model']:
        model.load_state_dict(torch.load(args.eval_model, map_location="cpu"))
        model.eval()
        eval_dir = "result/" + args.eval_model.split("/")[1] + "/eval/"


    # 条件付き生成用のconditionを作成
    # conditional_label = [[params.condition_values["Average path length"][i]] for i in range(3)]
    # conditional_label = torch.tensor(conditional_label)
    conditional_labels = utils.get_condition_values(params.condition_params, params.condition_values)
    
    # label normalization (trainデータのスケールで正規化)
    if params.label_normalize:
        datasets = joblib.load(params.twitter_train_path)
        train_label = datasets[1]
        # Search max, min
        max_val, min_val = -1, 10000
        for i, label in enumerate(train_label):
            if max_val < label.item():
                max_val = label.item()
            if min_val > label.item():
                min_val = label.item()
        # Normalization
        for i, data in enumerate(conditional_labels):
            conditional_labels[i] = (data - min_val) / (max_val - min_val)

    # 条件付き生成
    result_low = model.generate(
        params.number_of_generated_samples,
        torch.tensor([conditional_labels[0]]).float(),
        max_size=params.generate_edge_num
    )
    result_middle = model.generate(
        params.number_of_generated_samples,
        torch.tensor([conditional_labels[1]]).float(),
        max_size=params.generate_edge_num
    )
    result_high = model.generate(
        params.number_of_generated_samples,
        torch.tensor([conditional_labels[2]]).float(),
        max_size=params.generate_edge_num
    )
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
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args)))  # args，run_date，git_revisionなどを追加した辞書を取得
    
    if params.args['eval_model'] is None:
        print("評価対象のモデルを指定してください.")
        exit()
        
    # ログ設定
    logger = logging.getLogger(__name__)
    result_dir = params.args['eval_model'].split('/')[0] + '/' + params.args['eval_model'].split('/')[1]
    set_logging(result_dir, file_name="eval")  # ログを標準出力とファイルに出力するよう設定

    # eval
    eval(params, logger)
