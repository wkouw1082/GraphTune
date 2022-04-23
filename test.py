"""
Neural networkを使用したmodelをtestするためのモジュール.

A) ReEncoderの事前学習結果のtest
  1) test_re_encoder()
"""

import os
import argparse
import logging
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchinfo import summary
from torchviz import make_dot
import shutil
import joblib
import math

from config import common_args, Parameters
import utils
from utils import dump_params, setup_params
from utils import set_logging, make_dir, get_gpu_info
import preprocess as pp
from models import cvae, re_encoder


# TODO デバッグしながら確認する。
def test_re_encoder(params, args, logger):
	"""ReEncoder modelをtestする関数

		Args:
			params (config.Parameters)  : global変数のset
			args   (argparse.Namespace) : コンソールの引数のset
			logger ()					: logging
	"""
	# device
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# 前処理後のdataをloadし, 加工する
	test_dataset = joblib.load("dataset/test/onehot")
	test_label = joblib.load("dataset/test/label")
	test_conditional = joblib.load("dataset/test/conditional")
	time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")

	test_label = [element.to(device) for element in test_label]

	dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
	dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]
	test_conditional = torch.cat([test_conditional for _  in range(test_dataset.shape[1])],dim=1).unsqueeze(2)

	logger.info(f"\n--------------\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n")

	# Dataloader作成
	test_data_num = test_dataset.shape[0]
	test_label_args = torch.LongTensor(list(range(test_data_num)))
	test_dl = DataLoader(
			TensorDataset(test_label_args, test_dataset),\
			shuffle=True, batch_size=params.model_params["batch_size"])

	# モデル選択
	model = re_encoder.ReEncoder(dfs_size-1, params, device)
	if args.re_encoder_file:	# モデルをload
		model.load_state_dict(torch.load(args.re_encoder_file, map_location="cpu"))
	else:
		logger.info("モデルが選択されていません.")
		exit()
	model = model.to(device)	# modelをGPUに乗せる
	logger.info("モデル概要")
	logger.info(summary(model, input_size=(params.model_params["batch_size"], test_dataset.shape[1], dfs_size-1), col_names=["output_size", "num_params"]))

	# test開始
	logger.info("Start test ...")
	model.eval()
	## データをイレテー卜する
	for i, (indicies, data) in enumerate(test_dl, 1):
		data = data.to(device)

		# Forward propagate
		graph_property = model(data)

	logger.info("test complete!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser = common_args(parser)
	parser.add_argument('--re_encoder_file', help="事前学習済みReEncoderの重みへのPATH")  	# 事前学習されたReEncoderの重みへのPATHを指定
	args = parser.parse_args()
	params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得

	# ログ設定
	logger = logging.getLogger(__name__)
	set_logging(result_dir)  # ログを標準出力とファイルに出力するよう設定

	# グローバル変数とargsをlog出力
	logger.info('parameters: ')
	logger.info(params)
	logger.info('args: ')
	logger.info(args)

	# test
	test_re_encoder(params, args, logger)