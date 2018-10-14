import tensorflow as tf


def random_augmentation(example, board_size):
    """Perform a random rotation/flip on the example.

    example["inputs"] needs to be of shape [game_length, 3, board_size, board_size].
    example["p_targets"] needs to be of shape [game_length] containing ints in [0, num_moves).
    example["legal_moves"] needs to be of shape [game_length, num_moves].

    1/8 chance to do on of:
    * do nothing
    * rotate 90° counter clockwise
    * rotate 180° counter clockwise
    * rotate 270° counter clockwise
    * flip along vertical axis
    * flip along horizontal axis
    * flip along diagonal axis from the upper left
    * flip along diagonal axis from the upper right

    Args:
        example: dict, go game
        board_size: int, board size
    Return:
        Randomly augmented example
    """
    flat_board = board_size * board_size
    num_moves = flat_board + 1

    inputs = example["inputs"]
    legal_moves = example["legal_moves"]
    p_targets = example["p_targets"]

    inputs = tf.convert_to_tensor(inputs, name='inputs')
    legal_moves = tf.convert_to_tensor(legal_moves, name='legal_moves')
    p_targets = tf.convert_to_tensor(p_targets, name='p_targets')

    rand_k = tf.random_uniform([], int(0), int(8), tf.int64, name="rand_k")

    for name, array in [["inputs", inputs], ["legal_moves", legal_moves], ["p_targets", p_targets]]:
        split = name != "inputs"
        split_p = name == "p_targets"

        if split:
            if split_p:
                # convert p_target index to one hot
                array = tf.one_hot(array, num_moves)

            # split array into a flat board and one int representing the pass move
            array, rest = tf.split(array, [flat_board, 1], 1)
            # reshape to boards
            array = tf.reshape(array, [-1, board_size, board_size])

            # need to transpose last 2 axes
            perm = [0, 2, 1]
        else:
            # need to transpose last 2 axes
            perm = [0, 1, 3, 2]

        scope = ""

        def _no_aug():
            nonlocal scope
            scope = "no_augmentation"
            return array

        def _rot90():
            nonlocal scope
            scope = "rot_90_counter"
            return tf.transpose(tf.reverse(array, [-1]), perm)

        def _rot180():
            nonlocal scope
            scope = "rot_180_counter"
            return tf.reverse(array, [-1, -2])

        def _rot270():
            nonlocal scope
            scope = "rot_270_counter"
            return tf.reverse(tf.transpose(array, perm), [-1])

        def _flip_left_right():
            nonlocal scope
            scope = "flip_left_right"
            return tf.reverse(array, [-1])

        def _flip_up_down():
            nonlocal scope
            scope = "flip_up_down"
            return tf.reverse(array, [-2])

        def _flip_diagonal_upper_left():
            nonlocal scope
            scope = "flip_diagonal_upper_left"
            return tf.transpose(array, perm)

        def _flip_diagonal_upper_right():
            nonlocal scope
            scope = "flip_diagonal_upper_right"
            return tf.transpose(tf.reverse(array, [-1, -2]), perm)

        cases = [
            (tf.equal(rand_k, 0), _no_aug),
            (tf.equal(rand_k, 1), _rot90),
            (tf.equal(rand_k, 2), _rot180),
            (tf.equal(rand_k, 3), _rot270),
            (tf.equal(rand_k, 4), _flip_left_right),
            (tf.equal(rand_k, 5), _flip_up_down),
            (tf.equal(rand_k, 6), _flip_diagonal_upper_left),
            (tf.equal(rand_k, 7), _flip_diagonal_upper_right)
        ]

        result = tf.case(cases, name=scope)

        if split:
            # reassemble the original shape from combined result tensor and rest tensor
            result = tf.reshape(result, [-1, flat_board])
            result = tf.concat([result, rest], 1)
            if split_p:
                result = tf.argmax(result, 1)

        example[name] = result

    return example


def split_exmaple_by_color(example):
    inputs = example["inputs"]
    legal_moves = example["legal_moves"]
    p_targets = example["p_targets"]
    v_targets = example["v_targets"]
    dataset_name = example["dataset_name"]

    mask_black = tf.equal(inputs[:, 2, 0, 0], 1)

    game_length = tf.boolean_mask(inputs, mask_black)
    game_length = tf.shape(game_length)[0]

    black_example = {
        "inputs": tf.boolean_mask(inputs, mask_black),
        "legal_moves": tf.boolean_mask(legal_moves, mask_black),
        "p_targets": tf.boolean_mask(p_targets, mask_black),
        "v_targets": tf.boolean_mask(v_targets, mask_black),
        "game_length": game_length,
        "dataset_name": dataset_name
    }

    mask_white = tf.not_equal(inputs[:, 2, 0, 0], 1)

    game_length = tf.boolean_mask(inputs, mask_white)
    game_length = tf.shape(game_length)[0]

    white_example = {
        "inputs": tf.boolean_mask(inputs, mask_white),
        "legal_moves": tf.boolean_mask(legal_moves, mask_white),
        "p_targets": tf.boolean_mask(p_targets, mask_white),
        "v_targets": tf.boolean_mask(v_targets, mask_white),
        "game_length": game_length,
        "dataset_name": dataset_name
    }

    return [black_example, white_example]


def mapping(elem):
    position, player = elem

    ones = tf.ones_like(position)
    zeros = tf.zeros_like(position)

    black_mask = tf.equal(position, 1)
    black_stones = tf.where(black_mask, ones, zeros)

    white_mask = tf.equal(position, -1)
    white_stones = tf.where(white_mask, ones, zeros)

    def _black():
        return tf.stack([black_stones, white_stones, ones], axis=0)

    def _white():
        return tf.stack([white_stones, black_stones, zeros], axis=0)

    cases = [(tf.equal(player, 1), _black),
             (tf.equal(player, -1), _white)]

    new_pos = tf.case(cases)

    return new_pos


def format_test(example, board_size):
    inputs = example["inputs"]
    game_length = example['game_length']

    winner = example.pop('winner')
    to_play = example.pop('to_play')
    to_play = tf.cast(to_play, tf.int64)

    new_inputs = tf.map_fn(mapping, (inputs, to_play), dtype=tf.int64, back_prop=False)
    new_inputs = tf.reshape(new_inputs, [-1, 3, board_size, board_size])
    example["inputs"] = new_inputs

    def _draw():
        return tf.zeros_like(game_length)

    def _not_draw():
        is_winner = tf.ones_like(to_play)
        not_winner = tf.negative(is_winner)
        return tf.where(tf.equal(to_play, winner), is_winner, not_winner)

    cases = [(tf.equal(winner, 0), _draw),
             (tf.not_equal(winner, 0), _not_draw)]

    v_targets = tf.case(cases)
    example["v_targets"] = v_targets

    return example


def format_example(example, hparams):
    inputs = example["inputs"]
    game_length = example['game_length']

    winner = example.pop('winner')
    to_play = example.pop('to_play')
    to_play = tf.cast(to_play, tf.int64)

    partitions = tf.range(hparams.max_length)

    _inputs = tf.dynamic_partition(inputs, partitions, hparams.max_length, "unstack_inputs")
    _to_play = tf.dynamic_partition(to_play, partitions, hparams.max_length, "unstack_to_play")

    new_inputs = []
    for position, player in zip(_inputs, _to_play):
        if position == [] and player == []:
            break
        position = tf.reshape(position, [hparams.board_size, hparams.board_size])
        player = tf.reshape(player, [])

        ones = tf.ones_like(position)
        zeros = tf.zeros_like(position)

        black_mask = tf.equal(position, 1)
        black_stones = tf.where(black_mask, ones, zeros)

        white_mask = tf.equal(position, -1)
        white_stones = tf.where(white_mask, ones, zeros)

        def _black():
            return tf.stack([black_stones, white_stones, ones], axis=0)

        def _white():
            return tf.stack([white_stones, black_stones, zeros], axis=0)

        cases = [(tf.equal(player, 1), _black),
                 (tf.equal(player, -1), _white)]

        new_pos = tf.case(cases)

        new_inputs.append(new_pos)

    new_inputs = tf.stack(new_inputs, axis=0)
    example["inputs"] = new_inputs

    def _draw():
        return tf.zeros_like(game_length)

    def _not_draw():
        is_winner = tf.ones_like(to_play)
        not_winner = tf.negative(is_winner)
        return tf.where(tf.equal(to_play, winner), is_winner, not_winner)

    cases = [(tf.equal(winner, 0), _draw),
             (tf.not_equal(winner, 0), _not_draw)]

    v_targets = tf.case(cases)
    example["v_targets"] = v_targets

    return example
