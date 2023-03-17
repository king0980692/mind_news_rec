# AIR tensorflow version
Tensorflow implementation for AIR model

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.11.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

## Description
* AIR.py: AIR model
* AIR_rel_u.py: Modification of AIR, each **user** has **its own behavior**
* AIR_rel_ui.py: Modification of AIR, each **user and item** has **its own behavior**
* Code 寫死了（只能用來跑 Beibei 和 Taobao 資料集 ）

The instruction of commands has been clearly stated in the codes (see the parser function in utility/parser.py).

* Example command
```
python AIR_rel_u.py --dataset Beibei --data_path Data/ --regs [1e-4] --embed_size 64 --lr 0.001 --batch_size 2048 --epoch 1000 --neg_num 4 --model_type air_rel_u
```
or
```sh run.sh ```

## Experiment results
* Parameters setting: 
  * Embedding size: 64
  * Learing rate: 0.001
  * Number of negative sampling: 4  
  * Regularizer: 1e-4
* The results of Recall@10 in both dataset with different batch size: 

| Beibei |  AIR   | AIR_rel_u  | AIR_rel_ui |
|  ----  | ----  | ----  | ----  |
| 256  | 0.03725  epoch 8 | 0.03435  epoch 3 | 0.02970  epoch 1 |
| 1024  | 0.04342  epoch 32 | 0.04250  epoch 7 | 0.02675  epoch 4 |
| 2048 | 0.04831  epoch 19 | 0.04812  epoch 13 | 0.03021  epoch 3 |

| Taobao |  AIR   | AIR_rel_u  | AIR_rel_ui |
|  ----  | ----  | ----  | ----  |
| 256  | 0.10671 epoch 19 | 0.05155 epoch 9 | 0.03370 epoch 3 |
| 1024  | 0.12622  epoch 58 | 0.06234 epoch 19 |0.04101 epoch 8 |
| 2048 | 0.13336  epoch 79 | 0.06763 epoch 24 | 0.04388 epoch 12 |

## Note
* Best result in [GHCF](https://chenchongthu.github.io/files/AAAI_GHCF.pdf) model:
  * Parameters: same as above with 256 batch size
  * Beibei: Recall@10 = 0.1922
  * Taobao: Recall@10 = 0.0807
