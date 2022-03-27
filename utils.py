"""便利な関数群"""
from __future__ import annotations  # Python 3.7, 3.8はこの記述が必要
import torch
import subprocess
import logging
import json
from datetime import datetime
import os
from dataclasses import asdict
from typing import Any
import glob
import numpy as np


def get_git_revision() -> str:
    """
    現在のGitのリビジョンを取得

        Returns:
             str: revision ID
    """
    cmd = "git rev-parse HEAD"
    revision = subprocess.check_output(cmd.split())  # 現在のコードのgitのリビジョンを取得
    return revision.decode()


def setup_params(args_dict: dict[str, Any], path: str = None) -> dict[str, Any]:
    """
    コマンドライン引数などの辞書を受け取り，実行時刻，Gitのリビジョン，jsonファイルからの引数と結合した辞書を返す．
    
        Args:
            args_dict (dict): argparseのコマンドライン引数などから受け取る辞書
            path (str, optional): パラメータが記述されたjsonファイルのパス

        Returns:
            dict: args_dictと実行時刻，Gitのリビジョン，jsonファイルからの引数が結合された辞書．
                構造は {'args': args_dict, 'git_revision': <revision ID>, 'run_date': <実行時刻>, ...}．
    """
    run_date = datetime.now()
    git_revision = get_git_revision()  # Gitのリビジョンを取得

    param_dict = {}
    if path:
        param_dict = json.load(open(path, 'r'))  # jsonからパラメータを取得
    param_dict.update({'args': args_dict})  # コマンドライン引数を上書き
    param_dict.update({'run_date': run_date.strftime('%Y%m%d_%H%M%S')})  # 実行時刻を上書き
    param_dict.update({'git_revision': git_revision})  # Gitリビジョンを上書き
    return param_dict


def dump_params(params: 'config.Parameters', outdir: str, partial: bool = False) -> None:
    """
    データクラスで定義されたパラメータをjson出力する関数
    
    Args:
        params (:ogj: `Parameters`): パラメータを格納したデータクラス
        outdir (str): 出力先のディレクトリ
        partial (bool, optional): Trueの場合，args，run_date，git_revision を出力しない，
    """
    params_dict = asdict(params)  # デフォルトパラメータを取得
    if os.path.exists(f'{outdir}/parameters.json'):
        raise Exception('"parameters.json" is already exist. ')
    if partial:
        del params_dict['args']  # jsonからし指定しないキーを削除
        del params_dict['run_date']  # jsonからし指定しないキーを削
        del params_dict['git_revision']  # jsonからし指定しないキーを削
    with open(f'{outdir}/parameters.json', 'w') as f:
        json.dump(params_dict, f, indent=4)  # デフォルト設定をファイル出力


def set_logging(result_dir: str) -> 'logging.Logger':
    """
    ログを標準出力とファイルに書き出すよう設定する関数．

        Args:
            result_dir (str): ログの出力先
        Returns:
            設定済みのrootのlogger
    
        Example: 
        >>> logger = logging.getLogger(__name__)
        >>> set_logging(result_dir)
        >>> logger.info('log message...')
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # ログレベル
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # ログのフォーマット
    # 標準出力へのログ出力設定
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # 出力ログレベル
    handler.setFormatter(formatter)  # フォーマットを指定
    logger.addHandler(handler)
    # ファイル出力へのログ出力設定
    file_handler = logging.FileHandler(f'{result_dir}/log.log', 'w')  # ログ出力ファイル
    file_handler.setLevel(logging.DEBUG)  # 出力ログレベル
    file_handler.setFormatter(formatter)  # フォーマットを指定
    logger.addHandler(file_handler)
    return logger


def update_json(json_file: str, input_dict: dict[str, Any]) -> None:
    """jsonファイルをupdateするプログラム
        import json が必要

        Args:
            json_file (str): jsonファイルのpath
            input_dict (dict): 追加もしくは更新したいdict
    """
    with open(json_file) as f:
        df = json.load(f)

    df.update(input_dict)

    with open(json_file, 'w') as f:
        json.dump(df, f, indent=4)

def try_gpu(device, obj):
    """objectを指定されたdeviceに乗せる.

    Args:
        device (): device info
        obj (any): any object

    Returns:
        (any): 指定されたdeviceに乗ったobject
    """
    import torch
    return obj.to(device)
    # if torch.cuda.is_available():
    #     return obj.cuda(device)
    # return obj

def get_gpu_info(nvidia_smi_path: str = 'nvidia-smi', no_units: bool = True) -> str:
    """
    空いているgpuの番号を持ってくるプログラム

        Returns:
            str: 空いているgpu番号 or 'cpu'
    """
    keys = (
        'index',
        'uuid',
        'name',
        'timestamp',
        'memory.total',
        'memory.free',
        'memory.used',
        'utilization.gpu',
        'utilization.memory'
    )
    if torch.cuda.is_available():
        nu_opt = '' if not no_units else ',nounits'
        cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
        output = subprocess.check_output(cmd, shell=True)
        lines = output.decode().split('\n')
        lines = [line.strip() for line in lines if line.strip() != '']

        gpu_info = [{k: v for k, v in zip(keys, line.split(', '))} for line in lines]

        min_gpu_index = 0
        min_gpu_memory_used = 100
        for gpu in gpu_info:
            gpu_index = gpu['index']
            gpu_memory = int(gpu['utilization.gpu'])
            if min_gpu_memory_used >= gpu_memory:
                min_gpu_memory_used = gpu_memory
                min_gpu_index = int(gpu_index)

        return "cuda:" + str(min_gpu_index)
    else:
        return 'cpu'

def make_dir(required_dirs) :
    """入力されたディレクトリパスのリストから、ディレクトリを作成する関数

    既に存在しているディレクトリパスが指定された場合は、スキップされる.

        Args:
            required_dirs (list) : 作成したいディレクトリパスのリスト
    """
    dirs = glob.glob("*")
    for required_dir in required_dirs:
        if not required_dir in dirs:
            print("generate file in current dir...")
            print("+ "+required_dir)
            os.mkdir(required_dir)
        print("\n")

def convert2onehot(vec, dim):
    """
    特徴量のnumpy配列をonehotベクトルに変換
    
    Args:
        vec (): 特徴量のnumpy行列, int型 (サンプル数分の1次元行列)．
        dim (int): onehot vectorの次元
    
    Returns:
        (torch.Tensor) : onehot vectorのtensor行列
    """
    import torch
    return torch.Tensor(np.identity(dim)[vec])

def padding(vecs, flow_len, value=0):
    """flowの長さを最大flow長に合わせるためにzero padding
    
    Args:
        vecs (): flow数分のリスト. リストの各要素はflow長*特徴量長の二次元numpy配列
        flow_len (int): flow長
        value (int): paddingするvectorの要素値
    
    Returns:
        (): データ数*最大flow長*特徴量長の3次元配列
    """
    for i in range(len(vecs)):
        flow = vecs[i]
        if len(flow.shape)==2:
            diff_vec = np.ones((flow_len-flow.shape[0], flow.shape[1]))
        elif len(flow.shape) == 3:
            diff_vec = np.ones((flow_len-flow.shape[0], flow.shape[1], flow.shape[2]))
        else:
            diff_vec = np.ones(flow_len-flow.shape[0])
        diff_vec *= value
        vecs[i] = np.concatenate((flow, diff_vec), 0)
    return np.array(vecs)

def sample_dist(dist_tensor: 'torch.Tensor') -> 'torch.Tensor':
    """テンソルで表現された分布からサンプリングする
    
    Parameters
    ----------
    dist_tensor: torch.Tensor
        カテゴリカルな確率分布を表すテンソル (和が1である必要はない)
    Returns
    -------
        サンプリングした結果のカテゴリ
        
    Examples
    -------
        >>> dist = torch.Tensor([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]])
        >>> sample_dist(dist)
        tensor([[1],
                [0]])
    """
    categorical_obj = Categorical(dist_tensor)
    sample = categorical_obj.sample()
    return sample