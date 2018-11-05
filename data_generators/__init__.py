from data_generators.go_problem_19 import GoProblem19Cnn, GoProblem19Rnn
from data_generators.go_problem_19_small import GoProblem19SmallCnn, GoProblem19SmallRnn

__all__ = [
    "GoProblem19Cnn",
    "GoProblem19Rnn",
    "GoProblem19SmallCnn",
    "GoProblem19SmallRnn"
]


def get_problem_class(problem_name):
    if problem_name in __all__:
        return globals()[problem_name]
    else:
        raise Exception('The _problem name %s does not exist' % problem_name)
