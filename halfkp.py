import chess
import torch

NUM_SQ = 64
NUM_PT = 10
NUM_PLANES = (NUM_SQ * NUM_PT + 1)
INPUTS = NUM_PLANES * NUM_SQ # 41024

def orient(is_white_pov: bool, sq: int):
  return (63 * (not is_white_pov)) ^ sq

def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES

def get_halfkp_indices(board: chess.Board):
  def piece_indices(turn):
    indices = torch.zeros(INPUTS)
    for sq, p in board.piece_map().items():
      if p.piece_type == chess.KING:
        continue
      indices[halfkp_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
    return indices
  return (piece_indices(chess.WHITE), piece_indices(chess.BLACK))

def get_halfkp_indices_sparse(board: chess.Board):
  def piece_indices(turn):
    indices = torch.empty(INPUTS, layout=torch.sparse_coo)
    for sq, p in board.piece_map().items():
      if p.piece_type == chess.KING:
        continue
      indices[halfkp_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
    indices.coalesce()
    return indices
  return (piece_indices(chess.WHITE), piece_indices(chess.BLACK))
