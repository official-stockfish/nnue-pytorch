from .pp_3wide import PP3Wide


class RamRam(PP3Wide):
    """Every pair of rams on the board (a ram is a pawn blocked head-on by an enemy pawn), triangular index over
    the two ram ids (oriented square of the ram's lower pawn, ranks 2-6), oriented like the threat inputs."""

    HASH = 0x5E9FCAC8
    FEATURE_NAME = "Ram_Ram"
    INPUT_FEATURE_NAME = "Ram_Ram"
    MAX_ACTIVE_FEATURES = 8 * 7 // 2

    NUM_INPUTS = 40 * 39 // 2
    NUM_REAL_FEATURES = NUM_INPUTS
