import chess
import torch

NUM_SQ = 64
NUM_PT = 10
NUM_PLANES = (NUM_SQ * NUM_PT + 1)

def orient(is_white_pov: bool, sq: int):
  return (63 * (not is_white_pov)) ^ sq

def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES

class Features:
  name = 'HalfKP'
  inputs = NUM_PLANES * NUM_SQ # 41024

  def get_indices(self, board: chess.Board):
    def piece_indices(turn):
      indices = torch.zeros(inputs)
      for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
          continue
        indices[halfkp_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
      return indices
    return (piece_indices(chess.WHITE), piece_indices(chess.BLACK))

# Factors are used by the trainer to share weights between common elements of
# the features.
class Factorizer:
  name = 'HalfKPFactorized'
  factors = {
    'kings': 64,
    'pieces': 640,
  }
  inputs = sum(factors.values())

  def get_indices(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def coalesce_weights(self, weights):
    # incoming weights are in [256][INPUTS]
    print('factorized shape:', weights.shape)
    k_base = 41024
    p_base = k_base + 64
    result = []
    # Goal here is to add together all the weights that would be active for the
    # given piece and king position in the halfkp inputs.
    for i in range(k_base):
      k_idx = i // NUM_PLANES
      p_idx = i % NUM_PLANES
      w = weights.narrow(1, i, 1).clone()
      # King factorization.  Note that this is trickier than it appears.  The
      # weights for the king factor in the training set is equal to the number
      # of active base features.  This means that we (roughly) average the
      # number of pieces across the training dataset, and "bake" that in here.
      # If the king feature is treated as a one-hot encoding, you'd have to
      # divide by the average number of pieces in the dataset here (eg. divide
      # by 20).
      w = w + weights.narrow(1, k_base + k_idx, 1)
      # Piece factorization.  Note that p_idx 0 is not used by SF currently,
      # and the factorized pieces don't do the equivalent mapping, so that's
      # why we ignore it here.
      if p_idx > 0:
        w = w + weights.narrow(1, p_base + p_idx - 1, 1)
      result.append(w)
    return torch.cat(result, dim=1)

