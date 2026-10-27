# Experimental global threats and pawn structure

Default inputs are
`Global_Supports+Global_Attacks+Pawn_FrontierPairs+Pawn_FilePairs+K32Q2^`.
This is a new architecture requiring new training; old default networks and
checkpoints are incompatible. The serializer uses version `0x6a448afb`.

| Input | Logical rows | Exported rows | Hash |
| --- | ---: | ---: | --- |
| Global_Supports | 29,904 | 14,952 | 0x47535031 |
| Global_Attacks | 29,904 | 29,904 | 0x47415431 |
| Pawn_FrontierPairs | 16,807 | 16,807 | 0x50465031 |
| Pawn_FilePairs | 480 | 480 | 0x50464331 |
| K32Q2^ | 49,152 | 45,056 | 0x32b5e284 |

The file contains the shared int16 FT bias followed by these int8 tables in
order, then the existing 32 downstream stacks. L1=1024 and L2=L3=32 remain the
default architecture. Quantization uses symmetric [-127,127] FT weight bounds.

Friendly supports normalize their owner to White by reflecting Black's ranks,
without mirroring files. The learned table is `W`; virtual logical rows are
`cat(W,-W)`. White input selects a positive row for White supports and a negative
row for Black supports; Black input does the opposite. Gradients remain tied to
one table, and only that table is exported. No signed-index ABI or special CUDA
kernel is needed. Stockfish updates one physical signed support bank and adds
or subtracts it during activation assembly.

Enemy attacks retain unrestricted embeddings for both perspectives. Global
inputs swap colors and reflect ranks for Black, with no king-dependent file
mirroring; PSQ keeps its own orientation. Upstream threat target exclusions and
deduplication remain unchanged. Kings are not threat endpoints. Equal-type
contacts are visited once by descending physical square, then canonicalized
again after normalization. Per-color geometry block bases are
0,168,1848,4088,7672 for P,N,B,R,Q, with strides 84,336,560,896,1456.

Pawn frontiers use missing=0 and rank 2..7=1..6. File state is `7*white+black`,
using the most advanced pawn of each color. Boundary `f` indexes
`2401*f+49*state[f]+state[f+1]`; all seven boundaries activate, including empty
files. Reflection swaps colors and maps nonzero ranks to `7-rank`.

Within-file features select consecutive occupied pawn ranks regardless of color.
The 60 per-file states enumerate White-White rank pairs lexicographically, then
Black-Black, then White-Black distinct ranks lexicographically. Together with
frontiers this exposes complete pawn placement, including rear pawns.

`tests/test_global_features.py` contains an independent python-chess reference,
loader parity, signed gradients, round-trip serialization and permutation tests.
Set `GLOBAL_NNUE_TEST` to a compatible engine's `global-nnue-test` executable to
also check engine indices. Larger supplied position sets can be checked with:

```sh
python tests/test_global_features.py --positions /path/to/positions.epd \
  --engine-test /path/to/Stockfish/src/global-nnue-test
```

The CPU training smoke test pins the exact experimental Stockfish commit and
checks the full training, resume, serialization, FT optimization and evaluation
pipeline. A separate CPU compatibility workflow uses generated/random networks
and performs no training or optimizer steps. These tests do not establish
playing strength.
