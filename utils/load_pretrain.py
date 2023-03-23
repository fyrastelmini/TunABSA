import tensorflow as tf
from model.model import BiGRU_pretrain
def load_pretrain(model,checkpoint,freezing):
    print("Loading GRU pretraining checkpoint")
    pretrain = BiGRU_pretrain(vocab_size=model.vocab_size, embedding_dim=model.embedding_dim, gru_units=model.gru_units)
    pretrain.build_model()
    pretrain.model.load_weights(checkpoint)
    model.model.layers[1].set_weights(pretrain.model.layers[1].get_weights())
    model.model.layers[3].set_weights(pretrain.model.layers[3].get_weights())
    if freezing:
        print("GRU and embedding layers are frozen")
        model.model.layers[3].trainable=False
        model.model.layers[3].trainable=False
    return(model)