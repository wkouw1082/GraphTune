"""
可視化を実行するするためのインターフェースがまとめられたモジュール.
"""
import glob
import joblib
import shutil
import argparse
import os
import random

from config import common_args, Parameters
from utils import dump_params, setup_params
from utils import set_logging
import utils
from graph_process import graph_statistic
from graph_process import complex_networks
import bi


def graph_plot(params, args):
	"""グラフ特徴量をplotするための関数.

	Args:
		params (config.Parameters): configのグローバル変数のset
		args (): args
	"""
	calc_graph_statistic = graph_statistic.GraphStatistic()
	
	# eval.py で生成されたグラフの特徴量をcsvへ書き出し
	if args.eval_graphs:
		print("args.eval_graphs をチェック.")
		if args.eval_graphs[-1] != "/":
			print("パスの末尾に / を付けてください.")
			exit()
		print("生成されたグラフを読み込み.")
		visualize_dir = "result/" + args.eval_graphs.split("/")[1] + "/visualize/"
		graph_files = glob.glob(args.eval_graphs + "*")     # 生成されたグラフのパスのリスト
		for graph_file in graph_files:
			with open(graph_file, "rb") as f:
				graphs = joblib.load(f)
				# グラフオブジェクトから特徴量を計算し, csvファイルへ出力.
				calc_graph_statistic.graph2csv(graphs, csv_dir=visualize_dir+"csv/", file_name=graph_file.split("/")[-1], eval_params=params.eval_params) 
	else:
		# 新規に結果出力用ディレクトリを作成
		result_dir = f'result/{params.run_date}'
		required_dirs = [result_dir, result_dir+"/train", result_dir+"/eval", result_dir+"/visualize", result_dir+"/visualize/csv"]
		make_dir(required_dirs)
		visualize_dir = "result/" + params.run_date + "/visualize/"
	
	# データセットをcsvへ書き出し
	complex_network = complex_networks.ComplexNetworks()
	# datasets = complex_network.create_dataset(params.visualize_detail, do_type='visualize')
	datasets = complex_network.pickup_twitter_data(params.visualize_detail["twitter_pickup"][0])
	for (detail_name, detail) in params.visualize_detail.items():
		# calc_graph_statistic.graph2csv(datasets[detail_name][0], csv_dir=visualize_dir+"csv/", file_name=detail_name, eval_params=params.eval_params)
		calc_graph_statistic.graph2csv(datasets, csv_dir=visualize_dir+"csv/", file_name=detail_name, eval_params=params.eval_params)
	
	csv_paths = glob.glob(visualize_dir + "csv/*")
	csv_paths = sorted(csv_paths)
	
	# 各グラフ特徴量の平均値を計算する
	if os.path.isdir(visualize_dir + "average_param/"):
		shutil.rmtree(visualize_dir + "average_param")
	average_dir = visualize_dir + "average_param/"
	if os.path.isdir(visualize_dir + "percentile_param/"):
		shutil.rmtree(visualize_dir + "percentile_param")
	percentile_dir = visualize_dir + "percentile_param/"
	required_dirs = [average_dir, percentile_dir]
	utils.make_dir(required_dirs)
	for path in csv_paths:
		calc_graph_statistic.get_average_params(path, average_dir)
		calc_graph_statistic.get_percentile_params(path, percentile_dir)
	
	# 散布図を作成
	if os.path.isdir(visualize_dir + "scatter_diagram/"):
		shutil.rmtree(visualize_dir + "scatter_diagram")
	## csvのファイルパスからdir名を持ってくる
	dir_names = [os.path.splitext(os.path.basename(csv_path))[0] for csv_path in csv_paths]
	## dir名からdirを生成
	required_dirs = [visualize_dir + "scatter_diagram"] + [visualize_dir + "scatter_diagram/" + dir_name for dir_name in dir_names]
	utils.make_dir(required_dirs)
	for path in csv_paths:
		bi.scatter_diagram_visualize(params.eval_params, path, visualize_dir+"scatter_diagram/")
	
	# ヒストグラムを作成
	if os.path.isdir(visualize_dir + "histogram/"):
		shutil.rmtree(visualize_dir + "histogram")
	## csvのファイルパスからdir名を持ってくる
	dir_names = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]
	## dir名からdirを生成
	required_dirs = [visualize_dir + "histogram"] + [visualize_dir + "histogram/" + dir_name for dir_name in dir_names]
	utils.make_dir(required_dirs)
	for path in csv_paths:
		bi.histogram_visualize(params.eval_params, path, output_path=visualize_dir+"histogram/")
	
	# 散布図を結合
	if os.path.isdir(visualize_dir + "concat_scatter_diagram/"):
		shutil.rmtree(visualize_dir + "concat_scatter_diagram")
	dir_name = ''
	for index,path in enumerate(csv_paths):
		dir_name += os.path.splitext(os.path.basename(path))[0]
		if index != len(csv_paths)-1:
			dir_name += '&'
	## dir名からdirを生成
	required_dirs = [visualize_dir + "concat_scatter_diagram"] + [visualize_dir + "concat_scatter_diagram/" + dir_name]
	utils.make_dir(required_dirs)    
	bi.concat_scatter_diagram_visualize(params.eval_params, csv_paths, visualize_dir+"concat_scatter_diagram/"+dir_name+"/")

	# ヒストグラムを結合
	if os.path.isdir(visualize_dir + "concat_histogram/"):
		shutil.rmtree(visualize_dir + "concat_histogram")
	dir_name = ''
	for index,path in enumerate(csv_paths):
		dir_name += os.path.splitext(os.path.basename(path))[0]
		if index != len(csv_paths)-1:
			dir_name += '&'
	## dir名からdirを生成
	required_dirs = [visualize_dir + "concat_histogram"] + [visualize_dir + "concat_histogram/" + dir_name]
	utils.make_dir(required_dirs)
	bi.concat_histogram_visualize(params.eval_params, csv_paths, visualize_dir+"concat_histogram/"+dir_name+"/")

	# pair plotを作成
	if os.path.isdir(visualize_dir + "pair_plot/"):
		shutil.rmtree(visualize_dir + "pair_plot")
	required_dirs = [visualize_dir + "pair_plot"]
	utils.make_dir(required_dirs)
	bi.pair_plot(params.eval_params, csv_paths, visualize_dir+"pair_plot/")
	
	print("visualize complete!")

def graph_visualize(graph_path, result_dir=None, sampling_num=10):
	"""グラフを可視化する関数

	Args:
		graph_path (str): グラフデータまでのpathの正規表現
							e.g. graph_path = "result/20220329_171445/eval/*"
		result_dir (str, optional): 結果出力ディレクトリ. Defaults to None.
							e.g. result_dir = "result/20220329_171445/"
		sampling_num (int, optional): サンプリングする数. Defaults to 10.
	"""
	if result_dir is None:
		result_dir = graph_path.split("/")[0] + "/" + graph_path.split("/")[1] + "/"
	visualize_dir = result_dir + "visualize/"
	output_path = visualize_dir + "graph_structure/"
	if os.path.isdir(visualize_dir + "graph_structure/"):
		shutil.rmtree(visualize_dir + "graph_structure")
	required_dirs = [visualize_dir + "graph_structure"]
	utils.make_dir(required_dirs)
	# load graphs
	graph_files = glob.glob(graph_path)
	for graph_file in graph_files:
		with open(graph_file, "rb") as f:
			graphs = joblib.load(f)
			# sampling
			numbers = [i for i in range(0, len(graphs), 1)]
			indicies = random.sample(numbers, sampling_num)
			sampled_graphs = [graphs[i] for i in indicies]
			file_name_list = [f"{graph_file.split('/')[-1]}_{i}" for i in indicies]
			# graph visualize
			bi.graph_visualize(sampled_graphs, file_name_list, output_path)
	print("graph visualize complete!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser = common_args(parser)
	parser.add_argument('--eval_graphs')		# 生成されたグラフが保存されているディレクトリ
	args = parser.parse_args()
	params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得
 
	# visualize
	graph_plot(params, args)
	# graph_visualize(args.eval_graphs+"*")