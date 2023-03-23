import os
import argparse
import yaml

from transformers import BertTokenizer
from utils.dataloader import load_dataset, make_train_test_data, preprocess
import utils.evaluate as evaluate
from model.model import BiGRU_pretrain, BiGRU_attention


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True, help='Path to the config file')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--tokenizer_checkpoint', type=str, required=True, help='URL or path to the tokenizer checkpoint')
args = parser.parse_args()

# load configurations from the config file
with open(args.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# load the tokenizer
tokenizer = BertTokenizer.from_pretrained(args.tokenizer_checkpoint)
# you can define the tokens to remove
tokens_to_remove = [2,3]
# load the dataset
dataset = load_dataset(args.dataset_path,config['model_name'],tokenizer,True,tokens_to_remove)


# create and compile the model and prepare the data
if config['model_name'] == 'BiGRU_pretrain':
    model = BiGRU_pretrain(vocab_size=len(tokenizer.get_vocab()), embedding_dim=config['embedding_dim'], gru_units=config['gru_units'])
    model.build_model()
    model.compile_model(lr=config['learning_rate'])
    # preprocess dataset
    X_train, X_test, y_train_subject, y_test_subject, y_train_polarized, y_test_polarized=make_train_test_data(dataset,config['model_name'])
elif config['model_name'] == 'BiGRU_attention':
    model = BiGRU_attention(vocab_size=len(tokenizer.get_vocab()), embedding_dim=config['embedding_dim'], gru_units=config['gru_units'])
    model.build_model()
    model.compile_model(lr=config['learning_rate'])
    # preprocess dataset
    X_train, X_test, y_train, y_test=make_train_test_data(dataset,config['model_name'])
# train the model
if config['model_name'] == 'BiGRU_pretrain':
    model.train_model(X_train, y_train, batch_size=config['batch_size'], epochs=config['epochs'], validation_split=config['validation_split'])
elif config['model_name'] == 'BiGRU_attention':
    model.train_model(X_train,y_train, batch_size=config['batch_size'], epochs=config['epochs'], validation_split=config['validation_split'])

# test model
if config['model_name'] == 'BiGRU_pretrain':
    y_pred_subject, y_pred_polarized = model.predict(X_test)
    y_pred_subject=(y_pred_subject[:,:,0]> 0.5).astype(int)
    y_pred_polarized=(y_pred_polarized[:,:,0]> 0.5).astype(int)
    y_test=[y_test_subject,y_test_polarized]
    y_pred=[y_pred_subject,y_pred_polarized]
elif config['model_name'] == 'BiGRU_attention':
    y_pred = model.predict(X_test)
    y_pred = (y_pred[:,0,0]> 0.5).astype(int)
evaluate(y_test, y_pred,config['model_name'])
# save the model weights
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
model.model.save_weights(os.path.join('checkpoints', f'{config["model_name"]}.h5'))
