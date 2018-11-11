from models.go_models_cnn import AlphaZeroModel, AlphaZeroModelDense
from models.go_models_rnn import VanillaRNNModel, LSTMModel, GRUModel, \
    ConvRNNModel, MyConvRNNModel, ConvLSTMModel, MyConvLSTMModel, ConvGRUModel

__all__ = [
    # CNN
    "AlphaZeroModel",
    "AlphaZeroModelDense",
    # RNN
    "VanillaRNNModel",
    "LSTMModel",
    "GRUModel",
    # Conv RNN
    "ConvRNNModel",
    "MyConvRNNModel",
    "ConvLSTMModel",
    "MyConvLSTMModel",
    "ConvGRUModel",
]


def get_model_class(model_name):
    if model_name in __all__:
        return globals()[model_name]
    else:
        raise Exception('The _model name %s does not exist' % model_name)
