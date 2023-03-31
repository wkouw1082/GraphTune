# GraphTune: A Learning-Based Graph Generative Model With Tunable Structural Features

This repository is the official PyTorch implementation of GraphTune developed in NUT ComNets Lab.  
You can find the model descriptions and evaluations of GraphTune in [our paper](https://arxiv.org/abs/2201.11494). 

If you find this code useful in your research, please consider citing the following:

> Kohei Watabe, Shohei Nakazawa, Yoshiki Sato, Sho Tsugawa, and Kenji Nakagawa, ``GraphTune: A Learning-Based Graph Generative Model With Tunable Structural Features'', IEEE Transactions on Network Science and Engineering, 2023.

```bibtex
@ARTICLE{GraphTune,
  author={Watabe, Kohei and Nakazawa, Shohei and Sato, Yoshiki and Tsugawa, Sho and Nakagawa, Kenji},
  journal={IEEE Transactions on Network Science and Engineering}, 
  title={GraphTune: A Learning-Based Graph Generative Model With Tunable Structural Features}, 
  year={2023},
  volume={},
  number={},
  doi={10.1109/TNSE.2023.3244590}
}
```


## Requirement
- Python 3.7.6
- pytorch 1.9.0

## Installation
- Make a directory for preprocessed datasets. 
```shell
mkdir dataset
```
- Make a directory for output files. 
```shell
mkdir result
```
- Create virtual environments. 
```shell
python -m venv venv
```
- Install python modules. 
```shell
pip install --upgrade pip
pip install -r requirements.txt
```
- Install PyTorch. 
  - Refer [PyTorch official site](https://pytorch.org/get-started/previous-versions/).


## Training
- Train GraphTune model. 
```shell
python train.py --preprocess --use_model cvae
```
- Train GraphTune model in background. 
```shell
nohup python -u train.py --preprocess --use_model cvae &
```
- Retrain a trained model. 
```shell
nohup python -u train.py --use_model cvae --checkpoint_file result/{output-dir-name}/train/weight_35000 --init_epoch 35001 &
```

## Evaluation
- Generate graphs with a trained model. 
```shell
python eval.py --use_model cvae --eval_model result/{output-dir-name}/train/valid_best_weight
```

## Visualization
- Visualize generated graphs. 
```shell
python visualize.py --eval_graphs result/{output-dir-name}/eval/
```


## Parameter Settings
- You can set parameters in `config.py`. See comments in `config.py` for details.


## Contact
Kohei Watabe  
Associate Professor,  
Nagaoka University of Technology,  
Adress: Kamitomiokamachi 1603-1, Nagaoka Niigata 940-2188, Japan  
Tel: +81-258-47-9537  
E-mail: [k_watabe@vos.nagaokaut.ac.jp](k_watabe@vos.nagaokaut.ac.jp)  
Web: [NUT ComNets Lab.](https://kaede.nagaokaut.ac.jp/)
