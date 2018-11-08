from hparams.go_hparams_cnn import go_params_19_cnn, go_params_19_cnn_dense
from hparams.go_hparams_rnn import go_params_19_rnn, go_params_19_rnn_sorted, go_params_19_rnn_dense

__all__ = [
    "go_params_19_cnn",
    "go_params_19_cnn_dense",
    "go_params_19_rnn",
    "go_params_19_rnn_sorted",
    "go_params_19_rnn_dense"
]


def get_hparams(hparams_name):
    if hparams_name in __all__:
        return globals()[hparams_name]
    else:
        raise Exception('The hparams name %s does not exist' % hparams_name)
