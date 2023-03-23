import os
import argparse
import yaml
import tensorflow as tf
from transformers import AutoTokenizer
from utils.dataloader import load_dataset, preprocess
from models.models import BiGRU_pretrain, BiGRU_attention


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True, help='Path to the config file')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--tokenizer_checkpoint', type=str, required=True, help='URL or path to the tokenizer checkpoint')
args = parser.parse_args()

# load configurations from the config file
with open(args.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# load the dataset
dataset = load_dataset(args.dataset_path)

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint)

# preprocess the dataset
preprocessed_data = [] # replace with your preprocessing code
for example in dataset:
    preprocessed_data.append(preprocess(example, tokenizer)) 

# create and compile the model
if config['model_name'] == 'BiGRU_pretrain':
    model = BiGRU_pretrain(vocab_size=config['vocab_size'], embedding_dim=config['embedding_dim'], gru_units=config['gru_units'])
    model.build_model()
    model.compile_model(lr=config['learning_rate'])
elif config['model_name'] == 'BiGRU_attention':
    model = BiGRU_attention(vocab_size=config['vocab_size'], embedding_dim=config['embedding_dim'], gru_units=config['gru_units'])
    model.build_model(attention_units=config['attention_units'])
    model.compile_model(lr=config['learning_rate'])

# train the model
model.train_model(X_train, Y, batch_size=config['batch_size'], epochs=config['epochs'], validation_split=config['validation_split'])

# save the model weights
if not os.path.exists('models'):
    os.makedirs('models')
model.model.save_weights(os.path.join('models', f'{config["model_name"]}.h5'))
