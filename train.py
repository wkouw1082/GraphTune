"""
Neural networkを使用したmodelを学習するためのモジュール.
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
import copy

from config import common_args, Parameters
import utils
from utils import dump_params, setup_params
from utils import set_logging, make_dir, get_gpu_info
import preprocess as pp
from models import cvae, cvae_for_2_tuples, re_encoder, cvae_with_re_encoder


def train(params: 'config.Parameters', logger: 'logging.Logger'):
	"""汎用的なtrain関数

		Args:
			params (config.Parameters)  : global変数のset
			logger (logging.Logger)	    : logging
	"""
	# Seed値の固定
	# utils.fix_seed(params.seed)

	# Tensorboardの設定
	writer = SummaryWriter(log_dir=f"./result/{params.run_date}")

	# Open epoch毎lossが記載されるcsv file
	with open(f"./result/{params.run_date}/train/csv/train_loss_data.csv", "w") as csv_train_loss:
		csv_train_loss.write(f"epoch,loss\n")
	with open(f"./result/{params.run_date}/train/csv/valid_loss_data.csv", "w") as csv_valid_loss:
		csv_valid_loss.write(f"epoch,loss\n")

	# Open modelが推論した値と正解ラベルが記載されたcsv file
	with open(f"./result/{params.run_date}/train/csv/train_pred_data.csv", "w") as csv_train_pred_val:
		csv_train_pred_val.write(f"epoch,index,pred,correct\n")
	with open(f"./result/{params.run_date}/train/csv/valid_pred_data.csv", "w") as csv_valid_pred_val:
		csv_valid_pred_val.write(f"epoch,index,pred,correct\n")

	# deviceの設定
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# 前処理
	if params.args['preprocess']:
		logger.info("start preprocess...")
		shutil.rmtree("dataset")
		preprocess_dirs = ["dataset", "dataset/train", "dataset/valid", "dataset/test"]
		make_dir(preprocess_dirs)
		if params.args['preprocess_type'] == "dfs_2_tuples":
			## 2-tuplesのDFSコード
			pp.preprocess_for_2_tuples(params)
		else:
			## 5-tuplesのDFSコード
			pp.preprocess(params)

	# 前処理後のdataをload
	train_dataset = joblib.load("dataset/train/onehot")
	train_label = joblib.load("dataset/train/label")
	train_conditional = joblib.load("dataset/train/conditional")
	valid_dataset = joblib.load("dataset/valid/onehot")
	valid_label = joblib.load("dataset/valid/label")
	valid_conditional = joblib.load("dataset/valid/conditional")
	time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")

	# labelをdeviceに乗せる
	train_label = [element.to(device) for element in train_label]
	valid_label = [element.to(device) for element in valid_label]
	
	# datasetを作成
	use_model = params.args['use_model']
	if use_model == "cvae" or use_model == "cvae_with_re_encoder" or use_model == "re_encoder":
		dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
		dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]
		train_conditional = torch.cat([train_conditional for _ in range(train_dataset.shape[1])], dim=1).unsqueeze(2)
		valid_conditional = torch.cat([valid_conditional for _ in range(valid_dataset.shape[1])], dim=1).unsqueeze(2)
		if not use_model == "re_encoder":
			train_dataset = torch.cat((train_dataset, train_conditional), dim=2)
			valid_dataset = torch.cat((valid_dataset, valid_conditional), dim=2)
		logger.info(f"\n--------------\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n")
	elif use_model == "cvae_for_2_tuples":
		dfs_size = 2 * time_size + conditional_size
		train_conditional = torch.cat([train_conditional for _ in range(train_dataset.shape[1])], dim=1).unsqueeze(2)
		valid_conditional = torch.cat([valid_conditional for _ in range(valid_dataset.shape[1])], dim=1).unsqueeze(2)
		train_dataset = torch.cat((train_dataset, train_conditional), dim=2)
		valid_dataset = torch.cat((valid_dataset, valid_conditional), dim=2)
		logger.info(f"\n--------------\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n")
	else:
		logger.info("そのようなモデルは存在しません.")
		exit()

	# Dataloaderのseed固定
	# g = torch.Generator()
	# g.manual_seed(params.seed)

	# dataloaderを作成
	# train_data_num = train_dataset.shape[0]
	# train_label_args = torch.LongTensor(list(range(train_data_num)))
	# valid_data_num = valid_dataset.shape[0]
	# valid_label_args = torch.LongTensor(list(range(valid_data_num)))
	# train_dl = DataLoader(
	# 	TensorDataset(train_label_args, train_dataset),
	# 	shuffle=True,										# epoch毎にdataがshuffleされる
	# 	batch_size=params.model_params["batch_size"],		# mini-batchサイズ
	# 	drop_last=False,									# 指定したbacth_sizeでdataを割り切れなかった時、最後のバッチをdropしない
	# 	pin_memory=True,									# TensorをCUDAのpinされたメモリへコピーする
	# 	generator=g											# 乱数生成器を指定
	# )
	# valid_dl = DataLoader(
	# 	TensorDataset(valid_label_args, valid_dataset),
	# 	shuffle=False,
	# 	batch_size=params.model_params["batch_size"],
	# 	pin_memory=True,
	# 	generator=g
	# )
 
	# dataloaderを作成
	train_data_num = train_dataset.shape[0]
	train_label_args = torch.LongTensor(list(range(train_data_num)))
	valid_data_num = valid_dataset.shape[0]
	valid_label_args = torch.LongTensor(list(range(valid_data_num)))
	train_dl = DataLoader(
		TensorDataset(train_label_args, train_dataset),
		shuffle=True,										# epoch毎にdataがshuffleされる
		batch_size=params.model_params["batch_size"]		# mini-batchサイズ
	)
	valid_dl = DataLoader(
		TensorDataset(valid_label_args, valid_dataset),
		shuffle=False,
		batch_size=params.model_params["batch_size"]
	)

	# 学習するモデルの定義
	if use_model == "cvae":
		model = cvae.CVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
	elif use_model == "cvae_with_re_encoder":
		model = cvae_with_re_encoder.CVAEwithReEncoder(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
		if params.args['re_encoder_file']:
			model.re_encoder.load_state_dict(torch.load(args.re_encoder_file))
		else:
			logger.info("事前学習されたReEncoderの重みdataへのPATHを指定してください.")
			exit()
	elif use_model == "re_encoder":
		model = re_encoder.ReEncoder(dfs_size - 1, params, device)
	elif use_model == "cvae_for_2_tuples":
		model = cvae_for_2_tuples.CVAE(dfs_size=dfs_size, time_size=time_size,
										condition_size=conditional_size, params=params, device=device)
	
	# (Optional) チェックポイントをload
	if params.args['checkpoint_file'] and params.args['init_epoch']:
		model.load_state_dict(torch.load(params.args['checkpoint_file'], map_location="cpu"))
		init_epoch = int(params.args['init_epoch'])  # 初期エポック数
	else:
		init_epoch = 1

	# modelの概要をlogging
	model = model.to(device)  # modelをGPUに乗せる
	logger.info("モデル概要")
	logger.info(
		summary(
			model,
			input_size=(params.model_params["batch_size"], train_dataset.shape[1], train_dataset.shape[2]),
			col_names=["output_size", "num_params"],
			device=device
		)
	)

	# 最適関数の定義
	opt = optim.Adam(model.parameters(), lr=params.model_params["lr"])
	logger.info("最適化関数")
	logger.info(opt)


	# 学習開始
	logger.info("Start training ...")
	train_min_loss = 1e10  # trainでのmodel_lossの最小値
	valid_min_loss = 1e10  # validでのmodel_lossの最小値
	train_best_epoch = init_epoch  # trainでの最もmodel_lossが小さい時のepoch
	valid_best_epoch = init_epoch  # validでの最もmodel_lossが小さい時のepoch
	for epoch in range(init_epoch, params.epochs + 1, 1):
		logger.info("Epoch: [%d/%d]:" % (epoch, params.epochs))

		# 各epochにlossを0で初期化する
		if use_model == "cvae" or use_model == "cvae_for_2_tuples" or use_model == "cvae_with_re_encoder":
			encoder_loss_per_epoch = {"train": 0., "valid": 0.}
			five_tuples_dict = {"tu": 0., "tv": 0., "lu": 0., "lv": 0., "le": 0.}
			decoder_loss_per_epoch_dict = {"train": five_tuples_dict.copy(), "valid": five_tuples_dict.copy()}
			decoder_acc_per_epoch_dict  = {"train": five_tuples_dict.copy(), "valid": five_tuples_dict.copy()}
		if use_model == "cvae_with_re_encoder":
			re_encoder_loss_per_epoch = {"train": 0., "valid": 0.}
		model_loss_per_epoch = {"train": 0., "valid": 0.}

		# 訓練phaseと検証phase
		for phase in ["train", "valid"]:
			if phase == "train":
				model.train()  # modelを訓練するモードに設定する
				dataloader = train_dl  # 訓練用のDataLoaderを指定する
				label = train_label  # 訓練用のラベルを指定する
				condition = train_conditional
				logger.info("  train:")
			else:
				model.eval()  # modelを評価するモードに設定する
				dataloader = valid_dl  # 検証用のDataLoaderを指定する
				label = valid_label  # 検証用のラベルを指定する
				condition = valid_conditional
				logger.info("  valid:")

			# データをイレテー卜する
			for i, (indicies, data) in enumerate(dataloader, 0):
				opt.zero_grad()  # パラメータの勾配をゼロにします
				data = data.to(device)

				# Forward propagate
				# 訓練の時だけ、履歴を保持する
				with torch.set_grad_enabled(phase == 'train'):
					if use_model == "cvae":
						mu, sigma, tu, tv, lu, lv, le = model(data)
					elif use_model == "cvae_with_re_encoder":
						mu, sigma, tu, tv, lu, lv, le, graph_property = model(data)
					elif use_model == "re_encoder":
						graph_property = model(data)
					elif use_model == "cvae_for_2_tuples":
						mu, sigma, tu, tv = model(data)

					# Calculate loss
					## Encoder loss
					if use_model == "cvae" or use_model == "cvae_for_2_tuples":
						encoder_loss = model.encoder.loss(mu, sigma)
						encoder_loss_per_epoch[phase] += encoder_loss.item()
					elif use_model == "cvae_with_re_encoder":
						encoder_loss = model.cvae.encoder.loss(mu, sigma)
						encoder_loss_per_epoch[phase] += encoder_loss.item()

					## Decoder(Reconstruction) loss
					if use_model == "cvae":
						results = {"tu": tu, "tv": tv, "lu": lu, "lv": lv, "le": le}
						targets = {"tu": label[0][indicies], "tv": label[1][indicies], "lu": label[2][indicies],
								"lv": label[3][indicies], "le": label[4][indicies]}
						decoder_loss_dict, decoder_total_loss = model.decoder.loss(results, targets)
						for (key, val) in decoder_loss_dict.items():
							decoder_loss_per_epoch_dict[phase][key] += val.item()
					elif use_model == "cvae_with_re_encoder":
						results = {"tu": tu, "tv": tv, "lu": lu, "lv": lv, "le": le}
						targets = {"tu": label[0][indicies], "tv": label[1][indicies], "lu": label[2][indicies],
								"lv": label[3][indicies], "le": label[4][indicies]}
						decoder_loss_dict, decoder_total_loss = model.cvae.decoder.loss(results, targets)
						for (key, val) in decoder_loss_dict.items():
							decoder_loss_per_epoch_dict[phase][key] += val.item()
					elif use_model == "cvae_for_2_tuples":
						results = {"tu": tu, "tv": tv}
						targets = {"tu": label[0][indicies], "tv": label[1][indicies]}
						decoder_loss_dict, decoder_total_loss = model.decoder.loss(results, targets)
						for (key, val) in decoder_loss_dict.items():
							decoder_loss_per_epoch_dict[phase][key] += val.item()
					
					## ReEncoder loss
					if use_model == "cvae_with_re_encoder":
						graph_property = torch.squeeze(graph_property, 1)
						target_re_encoder = condition[indicies].to(device)
						target_re_encoder = target_re_encoder.transpose(1, 0)[0]
						re_encoder_loss = model.re_encoder.loss(graph_property, target_re_encoder)
						re_encoder_loss_per_epoch[phase] += re_encoder_loss.item()

					## Model loss
					## model_lossは, 各lossのミニバッチのsumに, 対応するalphaなどの定数を乗じて, 和をとったものである.
					if use_model == "cvae" or use_model == "cvae_for_2_tuples":
						model_loss = model.loss(encoder_loss, decoder_total_loss)
						model_loss_per_epoch[phase] += model_loss.item()
					elif use_model == "cvae_with_re_encoder":
						model_loss = model.loss(encoder_loss, decoder_total_loss, re_encoder_loss)
						model_loss_per_epoch[phase] += model_loss.item()
					elif use_model == "re_encoder":
						graph_property = torch.squeeze(graph_property, 1)
						target_re_encoder = condition[indicies].to(device)
						target_re_encoder = target_re_encoder.transpose(1, 0)[0]
						model_loss = model.loss(graph_property, target_re_encoder)
						model_loss_per_epoch[phase] += model_loss.item()
					
					# Calculate accuracy
					if use_model == "cvae" or use_model == "cvae_for_2_tuples":
						acc_dict = model.decoder.accuracy(results, targets)
						for (key, score) in acc_dict.items():
							decoder_acc_per_epoch_dict[phase][key] += score
					elif use_model == "cvae_with_re_encoder":
						acc_dict = model.cvae.decoder.accuracy(results, targets)
						for (key, score) in acc_dict.items():
							decoder_acc_per_epoch_dict[phase][key] += score
					
					# Save pred, correct
					if use_model == "re_encoder":
						for out_index in range(0, graph_property.shape[0], 1):
							if phase == "train":
								with open(f"./result/{params.run_date}/train/csv/train_pred_data.csv",
										"a") as csv_train_pred_val:
									csv_train_pred_val.write(
										f"{epoch},{out_index + params.model_params['batch_size'] * i},{graph_property[out_index].item()},{target_re_encoder[out_index].item()}\n")
							else:
								with open(f"./result/{params.run_date}/train/csv/valid_pred_data.csv",
										"a") as csv_valid_pred_val:
									csv_valid_pred_val.write(
										f"{epoch},{out_index + params.model_params['batch_size'] * i},{graph_property[out_index].item()},{target_re_encoder[out_index].item()}\n")

					# lossでnan検知
					if torch.isnan(model_loss):
						logger.info(f"model lossにnanを検知.")
						## nanを検知した時点でのmodelとoptを保存する
						torch.save(model.state_dict(),
								   "result/" + params.run_date + "/train/nan_iter_" + str(i) + "_weight_" + str(
									   epoch))
						torch.save(opt.state_dict(),
								   "result/" + params.run_date + "/train/nan_opt_iter_" + str(i) + "_weight_" + str(
									   epoch))
						## nanを検知した時点の1 iter前のmodelとoptを保存する
						torch.save(prev_model.state_dict(), "result/" + params.run_date + "/train/prev_nan_iter_" + str(
							i) + "_weight_" + str(epoch))
						torch.save(prev_opt.state_dict(),
									"result/" + params.run_date + "/train/prev_nan_opt_iter_" + str(
									   i) + "_weight_" + str(epoch))
						## プログラム終了
						exit()
					else:
						## バックアップを保存する
						prev_model = copy.deepcopy(model)
						prev_opt = copy.deepcopy(opt)

					# 訓練の時だけ, 誤差逆伝搬 + 勾配クリッピング + オプティマイズする
					if phase == "train":
						model_loss.backward()
      					# torch.nn.utils.clip_grad_norm_(model.parameters(), params.model_params["clip_th"])
						opt.step()
						torch.nn.utils.clip_grad_norm_(model.parameters(), params.model_params["clip_th"])

			# model_lossから計算グラフを可視化
			# dot = make_dot(model_loss, params=dict(model.named_parameters()))
			# dot.render("model_loss_cvae_with_re_encoder_eval")

			# Save loss / acc
			if phase == "train":
				## train loss / acc
				if use_model == "cvae" or use_model == "cvae_with_re_encoder" or use_model == "cvae_for_2_tuples":
					### encoder loss
					writer.add_scalar(f"{phase}_loss/encoder_loss", encoder_loss_per_epoch[phase] / train_data_num, epoch)
					logger.info(f"    encoder_loss = {encoder_loss_per_epoch[phase] / train_data_num}")
					### decoder loss
					for (key, val) in decoder_loss_per_epoch_dict[phase].items():
						writer.add_scalar(f"{phase}_loss/{key}_loss", val / train_data_num, epoch)
						logger.info(f"    {key}_loss = {val / train_data_num}")
					### decoder acc
					for (key, val) in decoder_acc_per_epoch_dict[phase].items():
						writer.add_scalar(f"{phase}_acc/{key}_acc",
										val / math.ceil(train_data_num / params.model_params['batch_size']), epoch)
						logger.info(
							f"    {key}_acc = {val / math.ceil(train_data_num / params.model_params['batch_size'])}")
					### re_encoder loss
					if use_model == "cvae_with_re_encoder":
						writer.add_scalar(f"{phase}_loss/re_encoder_loss", re_encoder_loss_per_epoch[phase] / train_data_num, epoch)
						logger.info(f"    re_encoder_loss = {re_encoder_loss_per_epoch[phase] / train_data_num}")
				### model loss
				writer.add_scalar(f"{phase}_loss/model_loss", model_loss_per_epoch[phase] / train_data_num, epoch)
				logger.info(f"    model_loss = {model_loss_per_epoch[phase] / train_data_num}")
				with open(f"./result/{params.run_date}/train/csv/train_loss_data.csv", "a") as csv_train_loss:
					csv_train_loss.write(f"{epoch},{model_loss_per_epoch[phase] / train_data_num}\n")
			else:
				## valid loss / acc
				if use_model == "cvae" or use_model == "cvae_with_re_encoder" or use_model == "cvae_for_2_tuples":
					### encoder loss
					writer.add_scalar(f"{phase}_loss/encoder_loss", encoder_loss_per_epoch[phase] / valid_data_num, epoch)
					logger.info(f"    encoder_loss = {encoder_loss_per_epoch[phase] / valid_data_num}")
					### decoder loss
					for (key, val) in decoder_loss_per_epoch_dict[phase].items():
						writer.add_scalar(f"{phase}_loss/{key}_loss", val / valid_data_num, epoch)
						logger.info(f"    {key}_loss = {val / valid_data_num}")
					### decoder acc
					for (key, val) in decoder_acc_per_epoch_dict[phase].items():
						writer.add_scalar(f"{phase}_acc/{key}_acc",
										val / math.ceil(valid_data_num / params.model_params['batch_size']), epoch)
						logger.info(
							f"    {key}_acc = {val / math.ceil(valid_data_num / params.model_params['batch_size'])}")
					### re_encoder loss
					if use_model == "cvae_with_re_encoder":
						writer.add_scalar(f"{phase}_loss/re_encoder_loss", re_encoder_loss_per_epoch[phase] / valid_data_num, epoch)
						logger.info(f"    re_encoder_loss = {re_encoder_loss_per_epoch[phase] / valid_data_num}")
				### model loss
				writer.add_scalar(f"{phase}_loss/model_loss", model_loss_per_epoch[phase] / valid_data_num, epoch)
				logger.info(f"    model_loss = {model_loss_per_epoch[phase] / valid_data_num}")
				with open(f"./result/{params.run_date}/train/csv/valid_loss_data.csv", "a") as csv_valid_loss:
					csv_valid_loss.write(f"{epoch},{model_loss_per_epoch[phase] / valid_data_num}\n")

		# Save model at checkpoint
		if epoch % params.model_save_point == 0:
			torch.save(model.state_dict(), "result/" + params.run_date + "/train/weight_" + str(epoch))
			logger.info(f'Checkpoint: {epoch}')

		# Save train best model
		if model_loss_per_epoch["train"] < train_min_loss:
			train_best_epoch = epoch
			train_min_loss = model_loss_per_epoch["train"]
			torch.save(model.state_dict(), "result/" + params.run_date + "/train/train_best_weight")
			logger.info(f'  Update train best epoch: {epoch}')

		# Save valid best model
		if model_loss_per_epoch["valid"] < valid_min_loss:
			valid_best_epoch = epoch
			valid_min_loss = model_loss_per_epoch["valid"]
			torch.save(model.state_dict(), "result/" + params.run_date + "/train/valid_best_weight")
			logger.info(f'  Update valid best epoch: {epoch}')

	writer.close()
	logger.info(f"train best epoch : {train_best_epoch}")
	logger.info(f"valid best epoch : {valid_best_epoch}")
	logger.info("train complete!")



if __name__ == "__main__":
	# 引数やGlobal変数を設定
	parser = argparse.ArgumentParser()
	parser = common_args(parser)
	args = parser.parse_args()
	try:
		params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得
	except AttributeError as e:
		print(e)
		params = Parameters(**setup_params(vars(args)))  # args，run_date，git_revisionなどを追加した辞書を取得

	# 結果出力用ディレクトリの作成
	result_dir = f'result/{params.run_date}'
	required_dirs = [result_dir, result_dir + "/train", result_dir + "/train/csv", result_dir + "/eval",
					 result_dir + "/test", result_dir + "/visualize", result_dir + "/visualize/csv"]
	make_dir(required_dirs)
	dump_params(params, f'{result_dir}')  # パラメータを出力

	# ログ設定
	logger = logging.getLogger(__name__)
	set_logging(result_dir, file_name="train")  # ログを標準出力とファイルに出力するよう設定

	# グローバル変数をlog出力
	logger.info('parameters: ')
	logger.info(params)

	# train
	train(params=params, logger=logger)