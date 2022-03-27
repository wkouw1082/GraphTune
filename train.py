"""
Neural networkを使用したmodelを学習するためのモジュール.

A) CVAEと事前学習済みReEncoderを使用した学習
  1) ReEncoderモデルを事前学習する.
  2) train_cvae_with_pre_trained_re_encoder()

B) ReEncoderの事前学習
  1) train_re_encoder()
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


def train_cvae_with_pre_trained_re_encoder(params, args, logger):
	"""modelを学習する関数
	
 	#TODO 汎用的なtrain関数にするには、modelを1つだけ読み込み、フローズンパラメータ(モデルの特定のパラメータを固定化)を設定可能にする。
	#TODO 勾配クリッピングの値が適切か調査
	#TODO validがtrain_dlを参照している問題あり

		Args:
			params (config.Parameters)  : global変数のset
			args   (argparse.Namespace) : コンソールの引数のset
	"""
	# Tensorboardの設定
	writer = SummaryWriter(log_dir=f"./result/{params.run_date}")

	# device
	device = get_gpu_info()

	# 前処理
	if args.preprocess:
		logger.info("start preprocess...")
		shutil.rmtree("dataset")
		preprocess_dirs = ["dataset", "dataset/train", "dataset/valid"]
		make_dir(preprocess_dirs)
		pp.preprocess(params)

	# 前処理後のdataをloadし, 加工する
	train_dataset = joblib.load("dataset/train/onehot")
	train_label = joblib.load("dataset/train/label")
	train_conditional = joblib.load("dataset/train/conditional")
	valid_dataset = joblib.load("dataset/valid/onehot")
	valid_label = joblib.load("dataset/valid/label")
	valid_conditional = joblib.load("dataset/valid/conditional")
	time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")

	train_label = [element.to(device) for element in train_label]
	valid_label = [element.to(device) for element in valid_label]

	dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
	dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]
	train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1).unsqueeze(2)
	valid_conditional = torch.cat([valid_conditional for _  in range(valid_dataset.shape[1])],dim=1).unsqueeze(2)
	train_dataset = torch.cat((train_dataset,train_conditional),dim=2)
	valid_dataset = torch.cat((valid_dataset,valid_conditional),dim=2)

	logger.info(f"\n--------------\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n")

	# Dataloader作成
	train_data_num = train_dataset.shape[0]
	train_label_args = torch.LongTensor(list(range(train_data_num)))
	valid_data_num = valid_dataset.shape[0]
	valid_label_args = torch.LongTensor(list(range(valid_data_num)))
	train_dl = DataLoader(
			TensorDataset(train_label_args, train_dataset),\
			shuffle=True, batch_size=params.model_params["batch_size"])
	valid_dl = DataLoader(
			TensorDataset(valid_label_args, valid_dataset),\
			shuffle=False, batch_size=params.model_params["batch_size"])

	# モデル選択
	if args.use_model == "VAEwithReEncoder":
		# CVAE (学習対象)
		model = cvae.CVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device).to(device)
		# ReEncoder (事前学習済み)
		re_encoder_model = re_encoder.ReEncoder(dfs_size-1, params, device).to(device)
		# ReEncoderのネットワークのパラメータを固定する
		for param in re_encoder_model.parameters():
			param.requires_grad = False
		# ReEncoderは検証モードに設定
		re_encoder_model.eval()
		logger.info("モデル概要")
		logger.info(summary(model, input_size=(params.model_params["batch_size"], train_dataset.shape[1], train_dataset.shape[2]), col_names=["output_size", "num_params"]))
		logger.info(summary(re_encoder_model, input_size=(params.model_params["batch_size"], train_dataset.shape[1], dfs_size-1), col_names=["output_size", "num_params"]))
	else:
		logger.info("モデルが選択されていません！！")
		exit()

	# 最適化関数の定義
	opt = optim.Adam(model.parameters(), lr=0.001)
	logger.info("最適化関数")
	logger.info(opt)

	# 学習開始
	logger.info("Start training ...")
	train_min_loss = 1e10   # model_lossの最小値
	best_epoch = 1			# 最もmodel_lossが小さい時のepoch
	for epoch in range(1, params.epochs+1, 1):
		logger.info("Epoch: [%d/%d]:"%(epoch, params.epochs))

		# 各epochにlossを0で初期化する
		encoder_loss_per_epoch = {"train": 0., "valid": 0.}
		five_tuples_dict = {"tu":0., "tv":0., "lu":0., "lv":0., "le":0.}
		decoder_loss_per_epoch_dict = {"train": five_tuples_dict.copy(), "valid": five_tuples_dict.copy()}
		decoder_acc_per_epoch_dict = {"train": five_tuples_dict.copy(), "valid": five_tuples_dict.copy()}
		re_encoder_loss_per_epoch = {"train": 0., "valid": 0.}
		model_loss_per_epoch = {"train": 0., "valid": 0.}

		# 訓練phaseと検証phase
		for phase in ["train", "valid"]:
			if phase == "train":
				model.train()   # modelを訓練するモードに設定する
			else:
				model.eval()    # modelを評価するモードに設定する

			# データをイレテー卜する
			for i, (indicies, data) in enumerate(train_dl, 1):
				opt.zero_grad()   # パラメータの勾配をゼロにします
				data = data.to(device)

				# Forward propagate
				# 訓練の時だけ、履歴を保持する
				with torch.set_grad_enabled(phase == 'train'):
					mu, sigma, tu, tv, lu, lv ,le = model(data.to(device), word_drop=params.word_drop_rate)
					graph_property = re_encoder_model(torch.cat([tu, tv, lu, lv, le], dim=2).to(device))

					# Calculate loss
					## Encoder loss
					encoder_loss = model.encoder.loss(mu, sigma)
					encoder_loss_per_epoch[phase] += encoder_loss.item()

					## Decoder(Reconstruction) loss
					results = {"tu":tu, "tv":tv, "lu":lu, "lv":lv, "le":le}
					targets = {"tu":train_label[0][indicies], "tv":train_label[1][indicies], "lu":train_label[2][indicies], "lv":train_label[3][indicies], "le":train_label[4][indicies]}
					decoder_loss_dict, decoder_total_loss = model.decoder.loss(results, targets)
					for (key, val) in decoder_loss_dict.items():
						decoder_loss_per_epoch_dict[phase][key] += val.item()

					## ReEncoder loss
					graph_property = torch.squeeze(graph_property, 1)
					target_re_encoder = train_conditional[indicies].to(device)
					target_re_encoder = target_re_encoder.transpose(1, 0)[0]
					re_encoder_loss = re_encoder_model.loss(graph_property, target_re_encoder)
					re_encoder_loss_per_epoch[phase] += re_encoder_loss.item()

					## Model loss
					## model_lossは, 各lossのミニバッチのsumに, 対応するalphaなどの定数を乗じて, 和をとったものである.
					model_loss =  model.loss(encoder_loss, decoder_total_loss) + params.model_params["gamma"] * re_encoder_loss.item()
					model_loss_per_epoch[phase] += model_loss.item()

					# Calculate accuracy
					acc_dict = model.decoder.accuracy(results, targets)
					for (key, score) in acc_dict.items():
						decoder_acc_per_epoch_dict[phase][key] += score

					# 訓練の時だけ, 誤差逆伝搬 + オプティマイズする
					if phase == "train":
						model_loss.backward()
						opt.step()

						# クリッピング
						torch.nn.utils.clip_grad_norm_(model.parameters(), params.model_params["clip_th"])
						# model_lossから計算グラフを可視化
						# dot = make_dot(model_loss, params=dict(model.named_parameters()))
						# dot.render("model_loss_cvae_with_re_encoder_eval")

			# Save loss/acc
			if phase == "train":
				writer.add_scalar(f"{phase}_loss/encoder_loss", encoder_loss_per_epoch[phase] / train_data_num, epoch)
				for (key, val) in decoder_loss_per_epoch_dict[phase].items():
						writer.add_scalar(f"{phase}_loss/{key}_loss", val / train_data_num, epoch)
				for (key, val) in decoder_acc_per_epoch_dict[phase].items():
					writer.add_scalar(f"{phase}_acc/{key}_acc", val / math.ceil(train_data_num / params.model_params["batch_size"]), epoch)
				writer.add_scalar(f"{phase}_loss/re_encoder_loss", re_encoder_loss_per_epoch[phase] / train_data_num, epoch)
				writer.add_scalar(f"{phase}_loss/model_loss", model_loss_per_epoch[phase] / train_data_num, epoch)
			else:
				writer.add_scalar(f"{phase}_loss/encoder_loss", encoder_loss_per_epoch[phase] / valid_data_num, epoch)
				for (key, val) in decoder_loss_per_epoch_dict[phase].items():
						writer.add_scalar(f"{phase}_loss/{key}_loss", val / valid_data_num, epoch)
				for (key, val) in decoder_acc_per_epoch_dict[phase].items():
					writer.add_scalar(f"{phase}_acc/{key}_acc", val / math.ceil(valid_data_num / params.model_params["batch_size"]), epoch)
				writer.add_scalar(f"{phase}_loss/re_encoder_loss", re_encoder_loss_per_epoch[phase] / valid_data_num, epoch)
				writer.add_scalar(f"{phase}_loss/model_loss", model_loss_per_epoch[phase] / valid_data_num, epoch)

		# Save model at checkpoint
		if epoch % params.model_save_point == 0:
			torch.save(model.state_dict(), "result/" + params.run_date + "/train/weight_" + str(epoch))
			logger.info(f'Checkpoint: {epoch}')

		# Save best model
		if model_loss_per_epoch["train"] < train_min_loss:
			best_epoch = epoch
			train_min_loss = model_loss_per_epoch["train"]
			torch.save(model.state_dict(), "result/" + params.run_date + "/train/best_weight")
			logger.info(f'Update best epoch: {epoch}')

	writer.close()
	logger.info(f"best epoch : {best_epoch}")
	logger.info("train complete!")


def train_re_encoder(params, args, logger):
	"""ReEncoder modelを学習する関数

		Args:
			params (config.Parameters)  : global変数のset
			args   (argparse.Namespace) : コンソールの引数のset
			logger ()					: logging
	"""
	# Tensorboardの設定
	writer = SummaryWriter(log_dir=f"./result/{params.run_date}")

	# device
	device = get_gpu_info()

	# 前処理
	if args.preprocess:
		logger.info("start preprocess...")
		shutil.rmtree("dataset")
		preprocess_dirs = ["dataset", "dataset/train", "dataset/valid"]
		make_dir(preprocess_dirs)
		pp.preprocess(params)

	# 前処理後のdataをloadし, 加工する
	train_dataset = joblib.load("dataset/train/onehot")
	train_label = joblib.load("dataset/train/label")
	train_conditional = joblib.load("dataset/train/conditional")
	valid_dataset = joblib.load("dataset/valid/onehot")
	valid_label = joblib.load("dataset/valid/label")
	valid_conditional = joblib.load("dataset/valid/conditional")
	time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")

	train_label = [element.to(device) for element in train_label]
	valid_label = [element.to(device) for element in valid_label]

	dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
	dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]
	train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1).unsqueeze(2)
	valid_conditional = torch.cat([valid_conditional for _  in range(valid_dataset.shape[1])],dim=1).unsqueeze(2)

	logger.info(f"\n--------------\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n")

	# Dataloader作成
	train_data_num = train_dataset.shape[0]
	train_label_args = torch.LongTensor(list(range(train_data_num)))
	valid_data_num = valid_dataset.shape[0]
	valid_label_args = torch.LongTensor(list(range(valid_data_num)))
	train_dl = DataLoader(
			TensorDataset(train_label_args, train_dataset),\
			shuffle=True, batch_size=params.model_params["batch_size"])
	valid_dl = DataLoader(
			TensorDataset(valid_label_args, valid_dataset),\
			shuffle=False, batch_size=params.model_params["batch_size"])

	# モデル選択
	# ReEncoder (事前学習済み)
	model = re_encoder.ReEncoder(dfs_size-1, params, device).to(device)
	logger.info("モデル概要")
	logger.info(summary(model, input_size=(params.model_params["batch_size"], train_dataset.shape[1], dfs_size-1), col_names=["output_size", "num_params"]))

	# 最適化関数の定義
	opt = optim.Adam(model.parameters(), lr=0.001)
	logger.info("最適化関数")
	logger.info(opt)

	# 学習開始
	logger.info("Start training ...")
	train_min_loss = 1e10   # model_lossの最小値
	best_epoch = 1			# 最もmodel_lossが小さい時のepoch
	for epoch in range(1, params.epochs+1, 1):
		logger.info("Epoch: [%d/%d]:"%(epoch, params.epochs))

		# 各epochにlossを0で初期化する
		model_loss_per_epoch = {"train": 0., "valid": 0.}

		# 訓練phaseと検証phase
		for phase in ["train", "valid"]:
			if phase == "train":
				model.train()   			# modelを訓練するモードに設定する
				dataloader = train_dl		# 訓練用のDataLoaderを指定する
				label = train_conditional	# 訓練用のラベルを指定する
			else:
				model.eval()    			# modelを評価するモードに設定する
				dataloader = valid_dl		# 検証用のDataLoaderを指定する
				label = valid_conditional	# 検証用のラベルを指定する

			# データをイレテー卜する
			for i, (indicies, data) in enumerate(dataloader, 1):
				opt.zero_grad()   # パラメータの勾配をゼロにします
				data = data.to(device)

				# Forward propagate
				# 訓練の時だけ、履歴を保持する
				with torch.set_grad_enabled(phase == 'train'):
					graph_property = model(data.to(device))

					# Calculate loss
     				## model_lossは, 各lossのミニバッチのsumに, 対応するalphaなどの定数を乗じて, 和をとったものである.
					graph_property = torch.squeeze(graph_property, 1)
					target_re_encoder = label[indicies].to(device)
					target_re_encoder = target_re_encoder.transpose(1, 0)[0]
					model_loss = model.loss(graph_property, target_re_encoder)
					model_loss_per_epoch[phase] += model_loss.item()

					# 訓練の時だけ, 誤差逆伝搬 + 勾配クリッピング + オプティマイズ(重みの更新)する
					if phase == "train":
						model_loss.backward()
      					# 勾配クリッピング(勾配爆発の抑止)
						# torch.nn.utils.clip_grad_norm_(model.parameters(), params.model_params["clip_th"])
						opt.step()
						# model_lossから計算グラフを可視化
						# dot = make_dot(model_loss, params=dict(model.named_parameters()))
						# dot.render("model_loss_cvae_with_re_encoder_eval")

			# Save loss/acc
			if phase == "train":
				writer.add_scalar(f"{phase}_loss/model_loss", model_loss_per_epoch[phase] / train_data_num, epoch)
			else:
				writer.add_scalar(f"{phase}_loss/model_loss", model_loss_per_epoch[phase] / valid_data_num, epoch)

		# Save model at checkpoint
		if epoch % params.model_save_point == 0:
			torch.save(model.state_dict(), "result/" + params.run_date + "/train/weight_" + str(epoch))
			logger.info(f'Checkpoint: {epoch}')

		# Save best model
		if model_loss_per_epoch["train"] < train_min_loss:
			best_epoch = epoch
			train_min_loss = model_loss_per_epoch["train"]
			torch.save(model.state_dict(), "result/" + params.run_date + "/train/best_weight")
			logger.info(f'Update best epoch: {epoch}')

	writer.close()
	logger.info(f"best epoch : {best_epoch}")
	logger.info("train complete!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser = common_args(parser)
	parser.add_argument('--use_model', default="VAEwithReEncoder")  # 学習するmodelのモジュール名
	parser.add_argument('--preprocess', action='store_true')        # 前処理をするかどうか
	args = parser.parse_args()
	params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得

	# 結果出力用ディレクトリの作成
	result_dir = f'result/{params.run_date}'  # 結果出力ディレクトリ
	required_dirs = [result_dir, result_dir+"/train", result_dir+"/eval", result_dir+"/visualize", result_dir+"/visualize/csv"]
	make_dir(required_dirs)
	dump_params(params, f'{result_dir}')  # パラメータを出力

	# ログ設定
	logger = logging.getLogger(__name__)
	set_logging(result_dir)  # ログを標準出力とファイルに出力するよう設定

	# グローバル変数とargsをlog出力
	logger.info('parameters: ')
	logger.info(params)
	logger.info('args: ')
	logger.info(args)

	# train
	# train_cvae_with_pre_trained_re_encoder(params, args, logger)
	train_re_encoder(params, args, logger)