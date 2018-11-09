from hparams.go_hparams_cnn import go_hparams_19_cnn
from hparams.go_hparams_rnn import go_hparams_19_rnn, go_hparams_19_rnn_sorted

__all__ = [
    "go_hparams_19_cnn",
    "go_hparams_19_rnn",
    "go_hparams_19_rnn_sorted",
]


def get_hparams(hparams_name):
    if hparams_name in __all__:
        return globals()[hparams_name]
    else:
        raise Exception('The hparams name %s does not exist' % hparams_name)
