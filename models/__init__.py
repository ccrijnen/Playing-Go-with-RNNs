from models.go_models_cnn import AlphaZeroModel, AlphaZeroModelDense
from models.go_models_rnn import ConvLSTMModel, MyConvLSTMModel, GRUModel

__all__ = [
    "AlphaZeroModel",
    "AlphaZeroModelDense",
    "ConvLSTMModel",
    "MyConvLSTMModel",
    "GRUModel"
]


def get_model_class(model_name):
    if model_name in __all__:
        return globals()[model_name]
    else:
        raise Exception('The _model name %s does not exist' % model_name)
