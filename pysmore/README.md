# Pysmore-torch

This repo refactor the smore project into pytoch version 


## Quick Start

### Install (python3.6+)
```bash
git clone git@github.com:cnclabs/pysmore.git
cd pysmore
pip install .
```

### Get Started

#### Split data 
```bash
./tools/split.sh ./test_data/ml-1m/ratings.dat ml-1m :: 0
```
This script will create a `exp` folder for temporarily stored the splited dataset
which will be accessed directly for training and evaluation.

and, the argument this script accepted by order is:
1. `data_path`: the path to the dataset
2. `data_name` : the data name defined for expeiment
3. `sep` : the seperator in dataset for time-order split data 
4. `header` : skip dataset header or not



#### Train & Eval


``` bash
# specify the gpu you want to use
./run_pysmore.sh torch_bpr ml-1m 1
```

This script accept three argument by order for training and caculate the score

1. `emb_name` : the name of saved embedding, which stored in exp folder
2. `data_name` : the name of data set name, which defined in previous script
3. `gpu` : using gpu or not, set 0  means using cpu for training 



## TODO 

- [ ] air-tf -> air-torch
- [ ] neg_sample time
- [ ] custom neg_sample input
