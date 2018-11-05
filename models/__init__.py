from models.go_models_cnn import AlphaZeroModel
from models.go_models_rnn import ConvLSTMModel, MyConvLSTMModel, LSTMModel

__all__ = [
    "AlphaZeroModel",
    "ConvLSTMModel",
    "MyConvLSTMModel",
    "LSTMModel"
]


def get_model_class(model_name):
    if model_name in __all__:
        return globals()[model_name]
    else:
        raise Exception('The _model name %s does not exist' % model_name)
