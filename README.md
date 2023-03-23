# TunABSA
This is the repository for the paper ["TUNIZI attention-based sentiment analysis with token-level"](https://openreview.net/pdf?id=jjRQfvptIg5)

Google colab demo: [Demo](https://colab.research.google.com/drive/1XhwOKeZVebuiNGtz_l2DpYylfVdcITzB?usp=sharing)
# Project Structure

This is the folder structure of the project:
```
README.md             # This file :)
requirements.txt      # pip requirements file
train.py              # model training script
...

dataset/             # datasets used for training
    TUNIZI_V2_FILTERED_CALIBRATED.csv     # TUNIZI_V2 dataset sample (sentencess with 10 or less words, calibrated 0/1 labels)
    dataset_out.csv                # our experimental TUNIZI dataset with token-level features
model/               # contains the different models used
    BiGRU_attention.yml   # embedding+bi-gru model configuration file
    BiGRU_pretrain.yml    # big-gru+attention model configuration file
    model.py              # models code
utils/            # useful scripts
    dataloader.py       # dataloading and preprocessing functions
    evaluate.py         # for metrics evaluations on test sets
    load_pretrain.py    # for loading trained embedding+bi-gru model layers into a big-gru+attention model

```

**Setting up the project environment**:

```shell script
# Clone the repo
git clone https://github.com/fyrastelmini/TunABSA.git
cd TunABSA

# Create a conda env
conda env create TunABSA

# Activate conda env
conda activate TunABSA

# Install requirements
pip install -r requirements.txt
```
# Training the models

## Pre-training the embedding+biGRU model on the 1k sentences dataset
* Run the following command-line:
```
python train.py --config_file "model/BiGRU_pretrain.yml" --dataset_path "dataset/dataset_out.csv" --tokenizer_checkpoint "ziedsb19/tunbert_zied"
```
## Pre-training the biGRU+attention model on the 60k sentences dataset
* Run the following command-line:
```
python train.py --config_file "model/BiGRU_attention.yml" --dataset_path "dataset/TUNIZI_V2_FILTERED_CALIBRATED.csv" --tokenizer_checkpoint "ziedsb19/tunbert_zied"
```
## Pre-training the biGRU+attention model on the 60k sentences dataset with a pre-trained embedding+Bi-GRU checkpoint
* Run the following command-line:
```
python train.py --config_file "model/BiGRU_attention.yml" --dataset_path "dataset/TUNIZI_V2_FILTERED_CALIBRATED.csv" --tokenizer_checkpoint "ziedsb19/tunbert_zied" --pretrain_checkpoint="checkpoints/BiGRU_pretrain.h5"
```
## Pre-training the biGRU+attention model on the 60k sentences dataset with a pre-trained embedding+Bi-GRU checkpoint and freezing on the pre-trained layers
* Run the following command-line:
```
python train.py --config_file "model/BiGRU_attention.yml" --dataset_path "dataset/TUNIZI_V2_FILTERED_CALIBRATED.csv" --tokenizer_checkpoint "ziedsb19/tunbert_zied" --pretrain_checkpoint="checkpoints/BiGRU_pretrain.h5" --freezing=True
```
