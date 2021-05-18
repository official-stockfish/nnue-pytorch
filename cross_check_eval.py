import argparse
import features
import serialize
import nnue_bin_dataset
import nnue_dataset
import subprocess
import re
from model import NNUE

def read_model(nnue_path, feature_set):
    with open(nnue_path, 'rb') as f:
        reader = serialize.NNUEReader(f, feature_set)
        return reader.model

def make_data_reader(data_path, feature_set):
    return nnue_bin_dataset.NNUEBinData(data_path, feature_set)

def eval_model_batch(model, batch):
    us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices = batch.contents.get_tensors('cuda')

    evals = [v.item() for v in model.forward(us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices) * 600.0]
    for i in range(len(evals)):
        if them[i] > 0.5:
            evals[i] = -evals[i]
    return evals

re_nnue_eval = re.compile(r'NNUE evaluation:\s*?(-?\d*?\.\d*)')

def compute_basic_eval_stats(evals):
    min_engine_eval = min(evals)
    max_engine_eval = max(evals)
    avg_engine_eval = sum(evals) / len(evals)
    avg_abs_engine_eval = sum(abs(v) for v in evals) / len(evals)

    return min_engine_eval, max_engine_eval, avg_engine_eval, avg_abs_engine_eval

def compute_correlation(engine_evals, model_evals):
    if len(engine_evals) != len(model_evals):
        raise Exception("number of engine evals doesn't match the number of model evals")

    min_engine_eval, max_engine_eval, avg_engine_eval, avg_abs_engine_eval = compute_basic_eval_stats(engine_evals)
    min_model_eval, max_model_eval, avg_model_eval, avg_abs_model_eval = compute_basic_eval_stats(model_evals)

    print('Min engine/model eval: {} / {}'.format(min_engine_eval, min_model_eval))
    print('Max engine/model eval: {} / {}'.format(max_engine_eval, max_model_eval))
    print('Avg engine/model eval: {} / {}'.format(avg_engine_eval, avg_model_eval))
    print('Avg abs engine/model eval: {} / {}'.format(avg_abs_engine_eval, avg_abs_model_eval))

    relative_model_error = sum(abs(model - engine) / (abs(engine)+0.001) for model, engine in zip(model_evals, engine_evals)) / len(engine_evals)
    relative_engine_error = sum(abs(model - engine) / (abs(model)+0.001) for model, engine in zip(model_evals, engine_evals)) / len(engine_evals)
    print('Relative engine error: {}'.format(relative_engine_error))
    print('Relative model error: {}'.format(relative_model_error))
    print('Avg abs difference: {}'.format(sum(abs(model - engine) for model, engine in zip(model_evals, engine_evals)) / len(engine_evals)))

def eval_engine_batch(engine_path, net_path, fens):
    engine = subprocess.Popen([engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    parts = ['uci', 'setoption name EvalFile value {}'.format(net_path)]
    for fen in fens:
        parts.append('position fen {}'.format(fen))
        parts.append('eval')
    parts.append('quit')
    query = '\n'.join(parts)
    out = engine.communicate(input=query)[0]
    evals = re.findall(re_nnue_eval, out)
    return [int(float(v)*208) for v in evals]

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--net", type=str, help="path to a .nnue net")
    parser.add_argument("--engine", type=str, help="path to stockfish")
    parser.add_argument("--data", type=str, help="path to .bin dataset")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint (used instead of nnue for local eval)")
    parser.add_argument("--count", type=int, default=100, help="number of datapoints to process")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    feature_set = features.get_feature_set_from_name(args.features)
    if args.checkpoint:
      model = NNUE.load_from_checkpoint(args.checkpoint, feature_set=feature_set)
    else:
      model = read_model(args.net, feature_set)
    model.eval()
    model.cuda()
    data_reader = make_data_reader(args.data, feature_set)

    fens = []
    results = []
    scores = []
    plies = []
    model_evals = []
    engine_evals = []
    i = -1

    def commit_batch():
        nonlocal fens
        nonlocal results
        nonlocal scores
        nonlocal plies
        nonlocal model_evals
        nonlocal engine_evals
        if len(fens) == 0:
            return
        b = nnue_dataset.make_sparse_batch_from_fens(feature_set, fens, scores, plies, results)
        model_evals += eval_model_batch(model, b)
        nnue_dataset.destroy_sparse_batch(b)
        engine_evals += eval_engine_batch(args.engine, args.net, fens)
        fens = []
        results = []
        scores = []
        plies = []

    done = 0
    while done < args.count:
        i += 1

        item = data_reader.get_raw(i)
        board = item[0]
        if board.is_check():
            continue

        fens.append(board.fen())
        results.append(int(round(item[2] * 2 - 1)))
        scores.append(int(item[3]))
        plies.append(1)

        done += 1

        if done % 1024 == 0:
            # don't do batches that are too big
            commit_batch()

    commit_batch()

    compute_correlation(engine_evals, model_evals)

if __name__ == '__main__':
    main()
