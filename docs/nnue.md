# NNUE

## Preface

This document explains the NNUE evaluation used by Stockfish and the ideas shared by other high-performance NNUE chess engines.

The main documentation is split into multiple parts.

[Basic theory](#basic-theory-a-starter-nnue) develops one starter NNUE from a piece-square table.
[Engineering](#engineering) provides a quantized reference implementation for inference, including practical details such as quantization and SIMD.
[Advanced topics: Bucketing](#advanced-topics-bucketing), [Advanced topics: Multiple layers](#advanced-topics-multiple-layers), and [Advanced topics: Feature set extensions](#advanced-topics-feature-set-extensions) cover additional techniques used by current top engines, with emphasis on Stockfish.

This document does not cover NNUE training workflows, datasets, or hyperparameters. For Stockfish-specific NNUE training, see the code in the rest of this repository (dedicated Stockfish training documentation may come in the future). For general NNUE training, see [Bullet](https://github.com/jw1912/bullet), a library designed specifically for training NNUE-style networks.

### Historical note

NNUE (ƎUИИ Efficiently Updatable Neural Network) is, broadly speaking, a neural network architecture that takes advantage of having minimal changes in the network inputs between subsequent evaluations. It was invented for Shogi by [Yu Nasu](https://www.chessprogramming.org/Yu_Nasu), integrated into [YaneuraOu](https://github.com/yaneurao/YaneuraOu) developed by Motohiro Isozaki in May 2018, and later ported to chess for use in Stockfish by [Hisayori Noda](https://www.chessprogramming.org/Hisayori_Noda) in June 2019, but is applicable to many other board games and perhaps even in other domains.

## Prerequisites

This is a guide to NNUE, not an introduction to machine learning. It assumes basic familiarity with vectors and matrices; matrix multiplication, activation functions, and loss functions; and the idea of gradient-based training.

Basic chess knowledge is also assumed.

The document introduces NNUE-specific concepts, including sparse input features, incremental updates, accumulators, and fixed-point inference, when they first become relevant.

## Table of contents

* [Preface](#preface)
    + [Historical note](#historical-note)
* [Prerequisites](#prerequisites)
* [Table of contents](#table-of-contents)
* [Basic theory: A starter NNUE](#basic-theory-a-starter-nnue)
    + [Evaluation functions](#evaluation-functions)
    + [Evaluation quality and speed](#evaluation-quality-and-speed)
    + [Normalized arithmetic cost](#normalized-arithmetic-cost)
    + [Piece-square tables](#piece-square-tables)
        - [Incremental PSQT updates](#incremental-psqt-updates)
    + [From PSQT to a starter NNUE](#from-psqt-to-a-starter-nnue)
        - [Starter input features](#starter-input-features)
        - [One perspective: the feature transformer](#one-perspective-the-feature-transformer)
        - [Incremental vector updates](#incremental-vector-updates)
        - [Squared Clipped ReLU](#squared-clipped-relu)
        - [Starter network diagram](#starter-network-diagram)
        - [Starter-network cost and capacity](#starter-network-cost-and-capacity)
            * [Starter-network arithmetic cost](#starter-network-arithmetic-cost)
            * [Further layers](#further-layers)
    + [Two perspectives and side-to-move evaluation](#two-perspectives-and-side-to-move-evaluation)
        - [Dual-perspective architecture and cost](#dual-perspective-architecture-and-cost)
* [Engineering](#engineering)
    + [Fixed-point integer quantization](#fixed-point-integer-quantization)
        - [Why integer quantization?](#why-integer-quantization)
        - [Reference scales](#reference-scales)
        - [Quantized CReLU and SCReLU](#quantized-crelu-and-screlu)
        - [Parameter and accumulator bounds](#parameter-and-accumulator-bounds)
    + [SIMD](#simd)
    + [Reference inference](#reference-inference)
        - [Feature transformer parameters](#feature-transformer-parameters)
        - [Perspective-relative feature indices](#perspective-relative-feature-indices)
        - [Accumulator](#accumulator)
            * [Refreshing an accumulator](#refreshing-an-accumulator)
            * [Updating an accumulator](#updating-an-accumulator)
        - [SCReLU and output layer](#screlu-and-output-layer)
        - [Fused SCReLU-output optimization](#fused-screlu-output-optimization)
* [Advanced topics: Bucketing](#advanced-topics-bucketing)
    + [Output bucketing](#output-bucketing)
    + [Input bucketing](#input-bucketing)
        - [Accumulator caches](#accumulator-caches)
    + [Horizontal mirroring](#horizontal-mirroring)
* [Advanced topics: Multiple layers](#advanced-topics-multiple-layers)
    + [The naive cost of dense layers](#the-naive-cost-of-dense-layers)
    + [8-bit-to-32-bit affine layers](#8-bit-to-32-bit-affine-layers)
    + [Pairwise-multiplication activation](#pairwise-multiplication-activation)
        - [Stockfish implementation note](#stockfish-implementation-note)
    + [Sparse inference](#sparse-inference)
        - [Four-value blocks](#four-value-blocks)
        - [Finding nonzero blocks](#finding-nonzero-blocks)
        - [Offline permutation for block sparsity](#offline-permutation-for-block-sparsity)
    + [Manual unrolling](#manual-unrolling)
* [Advanced topics: Feature set extensions](#advanced-topics-feature-set-extensions)
    + [Merged kings and HalfKAv2_hm](#merged-kings-and-halfkav2_hm)
    + [Full Threats feature set](#full-threats-feature-set)
        - [Deduplicating features](#deduplicating-features)
        - [I8 quantization for threat feature weights](#i8-quantization-for-threat-feature-weights)
    + [Pawn-pair features](#pawn-pair-features)
* [Miscellaneous](#miscellaneous)
    + [Deriving dual perspective from symmetry](#deriving-dual-perspective-from-symmetry)

## Basic theory: A starter NNUE

### Evaluation functions

A chess engine searches many possible continuations. At the end of each explored line, it needs an **evaluation function**: a fast function that estimates how favorable the resulting position is. We will begin with the conventional White-relative score, expressed in internal evaluation units: a positive score favors White and a negative score favors Black. This is a convenient starting convention; later, the NNUE will evaluate from the side-to-move perspective.

While traversing the search tree, the engine reaches each new position by making a move from a position on its search stack. Consecutive evaluations are therefore usually closely related, so reusing partial calculations can be quite valuable. What matters for now is that the engine must evaluate an enormous number of positions, so even a small cost per evaluation matters.

### Evaluation quality and speed

An engine's playing strength depends on both evaluation quality and search speed. Every engine must balance between quality of static evaluation and the number of positions it can search in a given time. A useful NNUE therefore tries to maximize evaluation quality per compute used.

### Normalized arithmetic cost

To compare architectures, this document uses a deliberately simplified **normalized arithmetic cost**. One 32-bit operation costs one unit, one 16-bit operation costs one-half unit, and one 8-bit operation costs one-quarter unit. This model provides decent estimations for the relative cost of vector operations, independent of SIMD width. It broadly accounts for moving values at those widths, but ignores cache behavior, memory-access patterns, and feature-index generation; those effects can be significant for advanced feature sets.

### Piece-square tables

A traditional evaluation function is a hand-written formula. One of its simplest useful components is a **piece-square table** (PSQT). A PSQT assigns a value to each combination of a piece and a square. For example, a white knight on d5 may receive a larger bonus than one on a1. In a White-relative PSQT, useful White-piece placements contribute positively and equivalent Black-piece placements contribute negatively. The score is the sum of the values associated with all pieces on the board:

```
score = value(white king on a1)
      + value(white pawn on c3)
      + value(black king on b8)
      + value(black rook on d4)
      + ...
```

This has the same mathematical form as a linear model. Represent the board as a long binary vector `x`: there is one input for every possible `(square, piece type, color)` combination, and that input is 1 exactly when the corresponding piece is present on that square. Put the PSQT values in a vector `w`. The same computation becomes:

```
score = dot(w, x) + bias
```

This weighted sum is the affine computation at the core of an artificial neuron. Neural-network libraries normally call `y = Ax + b` a Linear layer, although the bias makes the mathematical operation affine. A PSQT stops at the weighted sum and uses it directly as the score, so every feature contributes independently. A knight on d5 receives the same contribution regardless of whether that square is protected, whether the queens are present, or where the kings are. If the PSQT values are fitted from data, it is also a simple machine-learning model.

#### Incremental PSQT updates

A PSQT already has the efficient-update property that NNUE will use. Cache its score for the current position. When a pawn moves from c3 to c4, subtract the value for a white pawn on c3 and add the value for a white pawn on c4. A capture additionally subtracts the captured piece's value. The engine updates only the few terms affected by the move instead of summing every piece-square value again.

Assuming the cached score and PSQT entries are 32-bit values, each added or removed feature costs one normalized arithmetic unit. Updating `c3-c4` therefore costs two units, while `c3xd4` costs three. Recomputing the score from a position with 32 pieces would instead require roughly 32 such additions. This ignores table lookups and memory effects, but it captures the arithmetic saving that incremental evaluation makes possible.

In vector notation, this works because the features that changed are the only entries of `x` that changed. Updating `dot(w, x)` means adding or subtracting their corresponding weights. NNUE keeps exactly this property in its first layer; the difference is that each feature contributes a whole vector of learned values rather than one scalar PSQT value.

### From PSQT to a starter NNUE

The starter NNUE keeps the PSQT's sparse piece-square input and incremental-update property, but replaces its one scalar sum with a vector of sums. We will build that extension step by step.

The starter architecture is deliberately illustrative; later sections replace its simple features and dimensions with the current state of the art design. Quantization and low-precision inference are important implementation techniques, but are intentionally deferred until the basic computation is clear.

For readers familiar with machine learning, a PSQT acts like a perceptron's weighted sum over sparse binary inputs. The starter NNUE extends this single perceptron into a multilayer perceptron. Its defining engineering idea is the **feature transformer**: the sparse first layer whose pre-activation values can be incrementally updated instead of recomputed for every searched position.

#### Starter input features

The starter NNUE uses exactly the same binary piece-square features as the PSQT. We call this feature set **A** for "All pieces." A feature `(square, piece type, color)` is 1 when that piece occupies that square and 0 otherwise. With 64 squares, 6 piece types (pawn, knight, bishop, rook, queen, king), and 2 colors, there are `64 × 6 × 2 = 768` possible features.

Like the PSQT input, this vector is sparse: at most 32 features are active in a legal position. The engine stores the active feature indices rather than materializing all 768 values. An ordinary move removes one feature and adds one; captures and castling make additional changes, while a promotion adds the promoted piece's feature rather than a pawn feature.

The difference from the PSQT is not the input. A PSQT maps these features to one scalar score; the feature transformer maps them to a vector of learned PSQT-like values.

#### One perspective: the feature transformer

Start by replacing the PSQT's single weight vector with a weight matrix. Each row of the matrix is a learned PSQT-like table: summing its entries for the active piece-square features produces one value. With `N` rows, the feature transformer computes `N` such values at once:

```text
acc = W_ft x + b_ft
```

Up to this point, the computation is a collection of PSQT-like affine sums. The NNUE then applies an activation function and combines the resulting vector into one evaluation. This lets it model interactions that a single PSQT cannot.

#### Incremental vector updates

The PSQT cache was one scalar. The feature-transformer cache is the pre-activation accumulator vector, with one value for every first-layer output. Update it as follows:

```text
new_acc = old_acc
        - sum(W_ft[:, i] for removed feature indices i)
        + sum(W_ft[:, i] for added feature indices i)
```

This update applies to `acc`, not to the activated vector `z`. Apply the activation after the update, before passing the result to the output layer.

For a single move, the engine knows which starter piece-square features changed. Captures and promotions are simply additional removed or added features. This is the same update principle as for a PSQT, except that every feature contributes a weight vector instead of one scalar.

#### Squared Clipped ReLU

Now that we have a vector of PSQT-like values, we need to apply a nonlinear activation function.

Clipped ReLU (CReLU) bounds its input between 0 and 1:

```text
CReLU(x) = min(max(x, 0), 1)
```

Our activation function of choice is Squared Clipped ReLU (SCReLU), which squares CReLU:

```text
SCReLU(x) = CReLU(x)^2
```

Like other activation functions, SCReLU adds nonlinearity. Without it, consecutive affine layers could be collapsed into one affine transformation.

Bounding the activation range simplifies fixed-point scaling and efficient low-precision inference. Squaring the clipped value provides an additional nonlinear effect while preserving that bounded range. These are mathematical definitions; Engineering gives their integer implementation separately.

#### Starter network diagram

The starter evaluator uses the 768 piece-square features from White's fixed perspective. Its feature transformer produces `N` pre-activation values, which SCReLU converts into `N` inputs to the output layer. The network contains two affine layers:

```text
feature transformer:          A[768] -> N
activated output-layer input: N
output layer:                 N -> 1
```

The feature transformer is a dense matrix times a sparse vector. The output layer is evaluated in full for every position. The final output is a White-relative evaluation in internal units.

#### Starter-network cost and capacity

NNUEs are usually shallow. The feature transformer can be wide because its sparse input lets the engine update only a few weight columns after a move. After that point, however, the layers are dense and much more computationally expensive. The dense part of the network is therefore kept relatively narrow and shallow.

The starter architecture makes clear where the cost is paid. The feature transformer is refreshed from every active feature only when necessary, or updated from the few changed features after a move. The output layer is evaluated in full at every position. Production architectures choose their dimensions by balancing the evaluation quality gained against these costs.

##### Starter-network arithmetic cost

For the purposes of this estimation, we will assign bit widths reflecting practical usage, but treat them as implementation choices rather than part of the architecture; quantized inference is covered later in Engineering.

We assume 16-bit feature-transformer accumulators, a 16-bit-to-32-bit SCReLU activation, and a 32-bit output layer.

Let `N` be the feature-transformer width. A single changed feature column updates `N` 16-bit accumulator values, for an arithmetic cost of `N / 2`. Using an average estimate of `2.5` columns changed per evaluation, the work of the feature transformer is approximately `1.25N` units. This is a rough average: quiet moves change two columns and captures change three, while special moves can change more.

The fixed work after the feature transformer is approximately `3.5N` units:

| Work | Normalized cost |
| --- | ---: |
| Apply SCReLU to one `N`-value accumulator: two 16-bit clamps and one 16-bit square per value | `1.5N` |
| Dot product for the `N -> 1` output layer: `N` 32-bit products and `N` 32-bit accumulator additions | `2N` |

So, each evaluation in total costs roughly `5N` units.

##### Further layers

The starter network deliberately has no hidden dense layer. Inserting a naive `N -> M` hidden layer costs at minimum `MN` 8-bit products and additions, a term which quickly becomes the dominant cost even for small `M` values. This is why the starter architecture does not insert a further hidden layer. Advanced sections discuss the optimizations necessary to make further layers work in practice.

### Two perspectives and side-to-move evaluation

The starter evaluator uses a fixed White perspective and produces a White-relative score. To report a side-to-move-relative score, one could simply negate that score when Black is to move. This fixes the sign, but the network still sees the same White-oriented input in either case and therefore cannot learn effects that depend on whose turn it is.

A natural next attempt is to orient the input around the side to move. In White's view, White's pieces are "us" and Black's pieces are "them." In Black's view, the roles are reversed and the board is flipped vertically. For example, a White pawn on c3 is an "us pawn on c3" in White's view and a "them pawn on c6" in Black's view.

This gives the network a consistent "us, then them" representation, but one cached accumulator is no longer enough. The side to move changes after every move. If the engine stored only the side-to-move accumulator, its child position would use the opposite orientation; most feature indices would change, so the accumulator would need a full refresh instead of a small incremental update.

The solution is to maintain both orientations. The engine keeps a White-view accumulator and a Black-view accumulator, then updates each one incrementally after a move. This is the **dual-perspective** feature transformer used by strong NNUE engines.

The same feature-transformer weights are applied to both views. For a perspective `p`, its pre-activation **accumulator** is:

```text
acc_p = b_ft + sum(W_ft[:, i] for active feature indices i in view p)
z_p   = SCReLU(acc_p)
```

Once again, incremental updates only apply to `acc_p`. `z_p` is the activated value passed to the output layer; it cannot be updated with simple additions and subtractions because SCReLU is nonlinear.

At evaluation time, place the activated vector for the side to move first before passing the result to the output layer:

```text
White to move: dense_input = concat(z_white, z_black)
Black to move: dense_input = concat(z_black, z_white)
```

This ordering gives the output layer a consistent "us, then them" interpretation and exposes whose turn it is. The network is trained to produce a positive score when the side to move is favored; ordering the inputs alone does not impose that sign convention. If training data supplies a White-relative target `score_white`, use `score_white` when White is to move and `-score_white` when Black is to move.

#### Dual-perspective architecture and cost

Applying the starter feature transformer to both perspectives gives the following direct extension:

```text
shared feature transformer:  A[768] -> N, applied once per perspective
activated output-layer input: N + N = 2N
output layer:                 2N -> 1
```

For arithmetic-cost purposes, this dual-perspective evaluator costs the same as a single-perspective evaluator with a `2N`-wide feature transformer: both maintain `2N` accumulator values and pass `2N` activated values to the output layer.

## Engineering

Engineering presents the quantized reference inference path for the architecture from Basic theory. It explains the representation, update rules, and the operations that an optimized implementation accelerates with SIMD. This deliberately small reference evaluator is not a literal description of Stockfish's current production architecture; later sections introduce the extensions used by stronger engines.

### Fixed-point integer quantization

#### Why integer quantization?

Basic theory wrote network values as real numbers. A production NNUE instead performs inference with fixed-point integers: choose a scale `S`, store a value `x` as `q = round(Sx)`, and interpret it later as approximately `q / S`. The scale fixes the spacing between representable values, so every step has the same absolute size.

This is a particularly good fit for NNUE, which does not need a large magnitude range across many layers. Its shallow architecture and bounded activations let the implementation fix a useful range deliberately. What matters most is uniform precision within that range, which fixed-point integer quantization maximizes per bit.

Integer inference also has guaranteed identical behavior across all CPU architectures and compiler configurations, so long as the implementation avoids overflow. In contrast, the only way to achieve identical floating point behavior in practice is either to strictly stick to a scalar operation sequence, which defeats the point of vectorization, or explicitly not support certain CPU architectures.

#### Reference scales

The reference passes use `QA = 255` for the feature transformer and `QB = 64` for the output layer. Feature-transformer weights and biases are stored at scale `QA`, so the cached accumulator values are also at scale `QA`. SCReLU squares those values and therefore produces an intermediate at scale `QA²`. Output weights are stored at scale `QB`; divide the dot-product sum by `QA` before adding the output bias, which is stored at scale `QA × QB`.

| Value | Representation | Fixed-point scale |
| --- | --- | --- |
| Feature-transformer weight, bias, and accumulator | `int16` | `QA` |
| SCReLU result | `int32` | `QA²` |
| Output weight | `int16` | `QB` |
| Output bias | `int16` | `QA × QB` |
| Output score before final engine rescaling | `int64` | `QA × QB` |
| Final engine score | usually `int32` | engine-defined |

The quantized reference passes below use these representations directly. Quantization, converting a floating-point value `x` to integer scale `S` as `q = round(Sx)`, will typically be handled by the model trainer.

#### Quantized CReLU and SCReLU

The definitions in Basic theory operate on abstract real numbers. For an accumulator value `a_q` stored at scale `QA`, the reference implements the quantized version:

```text
CReLU_QA(a_q)  = clamp(a_q, 0, QA)
SCReLU_QA(a_q) = CReLU_QA(a_q)²
```

`CReLU_QA` represents a value at scale `QA`, and `SCReLU_QA` represents one at scale `QA²`. With `QA = 255`, the clipped value lies in `[0, 255]` and its square lies in `[0, 65025]`. The square therefore requires at least 16-bit unsigned or 32-bit signed storage in this reference implementation.

#### Parameter and accumulator bounds

A floating-point limit becomes a precise integer limit through `q = round(Sx)`. If `|x| <= L`, then `|q| <= round(SL)`. For a signed `b`-bit destination, choosing `L <= (2^(b - 1) - 1) / S` guarantees that the parameter itself fits its symmetric integer range. Tighter limits are often chosen for the arithmetic that follows, not merely for the storage type.

For the reference scales, common floating-point limits map as follows:

| Quantity | Scale | Float limit | Quantized limit |
| --- | ---: | ---: | ---: |
| Feature-transformer parameter | `QA = 255` | `+/-0.99` | `+/-252` |
| Output weight | `QB = 64` | `+/-1.99` | `+/-127` |
| Output bias | `QA * QB = 16320` | `+/-1.99` | `+/-32477` |

The first two limits leave considerable room in an `int16_t`; the output-weight limit also fits an `int8_t`. The output-bias limit is intentionally close to, but below, `int16_t`'s positive maximum of 32767. These are layer-specific examples, not universal NNUE limits: a network must derive its float clipping limits from its own scales and integer representations.

Storage limits alone do not prove inference safe. For every feature-transformer output `j`, the quantized accumulator must satisfy:

```text
-32768 <= bias_q[j] + sum(weight_q[feature][j] for active features) <= 32767
```

This must hold for every reachable position and for intermediate states of an incremental update. A very conservative sufficient condition is `abs(bias_q[j]), abs(weight_q[feature][j])) <= 504`. In practice, the accumulator will be very far from the signed 16-bit integer limits.

The output has a separate bound. With the usual output-weight limit `[-127, 127]`, one SCReLU product has magnitude at most `65025 * 127 = 8258175`, which fits in `int32_t`. A large single-layer NNUE can nevertheless theoretically overflow a 32-bit *sum* if every product reaches that magnitude with the same sign. In practice, trained networks do not approach that adversarial case; at typical output scales, reaching the signed 32-bit limit would correspond to an internal evaluation near `+/-200000`, far outside normal evaluation values. The reference uses `int64_t` for absolute safety. An optimized implementation should instead use `int32_t`.

### SIMD

SIMD (single instruction, multiple data) lets one CPU instruction apply the same operation to several adjacent values at once. NNUE is a particularly good fit: the inner loops are all vector operations.

The reference code deliberately uses scalar loops. With optimization enabled, a modern compiler can autovectorize the contiguous accumulator loops, and often much of the SCReLU and output dot product as well. This keeps the reference implementation readable while providing a useful baseline. Advanced topics cover further SIMD optimizations.

### Reference inference

The reference evaluator implements the dual-perspective extension of the starter architecture: `A[768] -> N`, maintained once for each perspective, followed by SCReLU and one `2N -> 1` output layer. Here and below, `N` is the width of each perspective's accumulator.

#### Feature transformer parameters

The feature transformer has one bias vector of length `N` and one `N`-value weight column for each input feature. Store each column contiguously because refreshing or updating an accumulator adds whole columns. The reference stores feature-transformer values and output-layer parameters as `int16_t`. Its output dot product uses an `int64_t` temporary, so the code makes its overflow behavior explicit; a production implementation may use `int32_t` only after verifying tighter bounds for its exported network.

```cpp
struct FeatureTransformer {
    int16_t weight[768][N];
    int16_t bias[N];
};

struct OutputLayer {
    int16_t weight[2 * N];
    int16_t bias;
};
```

#### Perspective-relative feature indices

The White-view feature index is the ordinary `(square, piece type, color)` encoding. To obtain the Black view, flip the board vertically and exchange the piece colors before applying the same encoding. Both perspectives can therefore share one feature-transformer weight matrix.

```cpp
int a_feature_index(
    Color     perspective,
    Square    square,
    PieceType piece,
    Color     color
) {
    if (perspective == Color::Black) {
        square = flip_vertical(square);
        color  = opposite(color);
    }

    return static_cast<int>(square)
         + 64 * (static_cast<int>(piece) + 6 * static_cast<int>(color));
}
```

This direct 768-entry encoding intentionally reserves some unreachable combinations, such as a pawn on the first or eighth rank. Keeping a uniform arithmetic index is simpler and faster than special-casing those entries.

#### Accumulator

The accumulator is position state stored on the search stack. It holds one `N`-value pre-activation vector for each perspective.

```cpp
struct NnueAccumulator {
    // Two vectors of size N: one for each perspective.
    int16_t v[2][N];

    int16_t* operator[](Color perspective) {
        return v[perspective];
    }

    const int16_t* operator[](Color perspective) const {
        return v[perspective];
    }
};
```

An accumulator can be updated eagerly as moves are made or lazily when evaluation needs it. In either case, the engine either refreshes it from all active features or derives it from its parent by applying the changed features.

##### Refreshing an accumulator

```cpp
void refresh_accumulator(
    const FeatureTransformer& transformer,
    NnueAccumulator&        new_acc,          // storage for the result
    const std::vector<int>& active_features,  // active features for this perspective
    Color                   perspective
) {
    for (int i = 0; i < N; ++i) {
        new_acc[perspective][i] = transformer.bias[i];
    }

    for (int a : active_features) {
        for (int i = 0; i < N; ++i) {
            new_acc[perspective][i] += transformer.weight[a][i];
        }
    }
}
```

##### Updating an accumulator

```cpp
void update_accumulator(
    const FeatureTransformer& transformer,
    NnueAccumulator&        new_acc,
    const NnueAccumulator&  prev_acc,
    const std::vector<int>& removed_features,
    const std::vector<int>& added_features,
    Color                   perspective
) {
    for (int i = 0; i < N; ++i) {
        new_acc[perspective][i] = prev_acc[perspective][i];
    }

    for (int r : removed_features) {
        for (int i = 0; i < N; ++i) {
            new_acc[perspective][i] -= transformer.weight[r][i];
        }
    }

    for (int a : added_features) {
        for (int i = 0; i < N; ++i) {
            new_acc[perspective][i] += transformer.weight[a][i];
        }
    }
}
```

#### SCReLU and output layer

After updating both accumulators, apply SCReLU and evaluate the direct `2N -> 1` output layer. The output-layer weights for the side to move (`stm`) occupy its first `N` inputs, followed by the non-side-to-move (`nstm`) inputs.

```cpp
constexpr int QA = 255;
constexpr int QB = 64;

int32_t screlu(int16_t x) {
    const int32_t clipped = std::clamp(
        static_cast<int32_t>(x), int32_t{0}, int32_t{QA}
    );
    return clipped * clipped;
}

int64_t evaluate(
    const OutputLayer&       output_layer,  // 2N -> 1
    const NnueAccumulator&   acc,
    Color                    stm
) {
    const Color nstm = opposite(stm);
    int64_t score = 0;

    for (int i = 0; i < N; ++i) {
        score += static_cast<int64_t>(screlu(acc[stm][i]))
               * output_layer.weight[i];
        score += static_cast<int64_t>(screlu(acc[nstm][i]))
               * output_layer.weight[N + i];
    }

    // SCReLU introduced one additional factor of QA.
    score /= QA;
    score += static_cast<int64_t>(output_layer.bias);
    return score;
}
```

The returned score is at scale `QA × QB`. At evaluation time, convert it to the engine's internal evaluation units as `round(raw_score * OutputScale / (QA * QB))`, where `OutputScale` is chosen by the engine. Specify the rounding rule as part of the evaluator so every supported target produces the same score. Model loading quantizes and bounds the parameters, while this final rescaling belongs to runtime evaluation.

The `int64_t` temporary gives the reference a generous output range. An optimized implementation can use 32-bit horizontal multiply-add after validating that its trained network remains safely within the practical range described above. This is the complete quantized forward pass for the reference architecture. It deliberately has no additional dense hidden layer.

#### Fused SCReLU-output optimization

Let `c = clamp(acc[stm][i], 0, 255)`. If output weights are bounded to `[-127, 127]`, then the intermediate `c * output_layer.weight[i]` lies in `[-32385, 32385]` and fits in a signed 16-bit integer.

This permits the reassociation:

```text
SCReLU(acc[stm][i]) * weight
= c² * weight
= c * (c * weight)
```

The straightforward evaluation first computes `c²` with a 16-bit × 16-bit -> 32-bit multiplication, then multiplies that 32-bit value by the weight. With the reassociation, the first multiplication is 16-bit × 16-bit -> 16-bit, and only the second is 16-bit × 16-bit -> 32-bit. Presenting this bounded intermediate explicitly can enable faster autovectorization. Alternatively, a manual SIMD implementation can guarantee the speed boost. The final product and output accumulation still require wider storage.

## Advanced topics: Bucketing

The starter NNUE uses one feature transformer and one output layer for every position. Strong engines can give the network more specialized parameters by selecting a small group of them from a cheap, discrete property of the position. This is called *bucketing*. The selected property must be easy to maintain during search, and the benefit must justify the additional model memory and complexity.

For readers familiar with mixture-of-experts models, buckets act like hand-selected experts: a deterministic chess rule chooses exactly one parameter set. Unlike a typical mixture-of-experts router, there is no learned gate and no weighted combination of experts.

### Output bucketing

Output bucketing specializes the final part of the evaluator without changing its input features or accumulators. Let `q(position)` choose one of `B` output buckets. The bucket selector should be cheap and deterministic. Piece count (`B = 8`) is a very common choice.

Instead of one output vector `w` and bias `beta`, keep one pair for each bucket:

```text
score = w[q(position)]^T concat(SCReLU(acc[stm]), SCReLU(acc[nstm]))
      + beta[q(position)]
```

Each bucket can therefore learn a different interpretation of the same accumulated features.

Only the selected output is evaluated, so output bucketing does not increase the direct `2N -> 1` arithmetic cost of a single evaluation. However, it does multiply the stored output parameters by `B`.

### Input bucketing

Input bucketing specializes the feature transformer itself. In chess NNUE, the usual selector is derived from the perspective's king square, so this technique is often called *king bucketing*. A bucket need not be unique to one king square: the selector may map several squares to the same bucket. Rather than sharing one PSQT-like weight column for every king location, the network gives a feature a separate column for its selected king bucket.

For perspective `p`, let `k_p(position)` be its input bucket. The accumulator becomes:

```text
acc[p] = bias[k_p(position)] + sum(W[k_p(position)][feature])
```

Equivalently, king bucketing upgrades the starter `(piece, square)` input feature to a `(king bucket, piece, square)` feature. With one bucket per square, this is a `(king square, piece, square)` feature; coarser schemes deliberately share columns among several king squares. Only the triples matching the current friendly king bucket are active. This view makes clear why a king move that changes buckets requires a refresh: it changes the king-bucket component of every active feature at once.

This lets the network interpret the same piece-square feature differently when the friendly king is elsewhere—for example, a pawn shield is meaningful only relative to its king.

In the reference implementation, this extension conceptually changes `weight[768][N]` and `bias[N]` into `weight[K][768][N]` and `bias[K][N]`, where `K` is the number of input buckets. The feature-index logic is unchanged after it has selected the bucket; accumulator maintenance chooses the corresponding slice.

The tradeoff is fundamental. With a fixed bucket, the usual incremental add/subtract update still works. When a king move changes the selected bucket, all active features refer to a different set of transformer weights, so the accumulator cannot be converted by a small delta and must be refreshed. Input bucketing also multiplies feature-transformer storage by the number of buckets. Since almost all of an NNUE's parameters lie in the feature transformer, input bucketing effectively multiplies the entire NNUE's total size.

A useful input-bucket selector changes relatively rarely along the search tree. Each transition can require a refresh, so a bucket based on a volatile property of the position can erase the advantage of incremental updates. King location is a useful selector partly because king moves are uncommon; coarser king buckets can reduce transitions further.

#### Accumulator caches

Refreshing is quite expensive, up to `max(num active features) * N` normalized cost units in the worst case scenario, which can be an order of magnitude more than a standard update. An accumulator cache reduces the cost of those unavoidable refreshes. It keeps a recent transformer sum and a snapshot of its active features for each perspective and input bucket. To refresh the accumulator for a position, the engine starts from the cache entry for the required bucket, subtracts columns for features absent from the new position, adds columns for newly present features, then records the resulting sum as the updated cache entry.

The cache is not valid merely because the bucket matches: its saved feature snapshot is what makes the difference update correct. On a cold entry, the cached sum is just the transformer bias and the refresh has the ordinary full cost. In search, however, nearby positions often share most pieces, so a per-thread cache can turn many bucket-changing refreshes into a short sequence of incremental updates. Stockfish calls this style of cache a *Finny table*.

### Horizontal mirroring

King-square features make horizontal mirroring straightforward. Choose one half of the board as canonical. For each perspective, if its friendly king lies on the other half, reflect every square horizontally before generating its `(king square, piece, square)` features. A position and its reflected counterpart then use the same canonical king bucket and the same feature indices. Equivalently, the columns for `(king square, piece, square)` and `(reflect(king square), piece, reflect(square))` are identical: they are the same stored column after canonicalization.

This shares parameters between left-right-related king configurations. With one canonical bucket per king square, it reduces the number of king buckets from 64 to 32 and approximately halves feature-transformer storage. It is a deliberate approximate symmetry: chess is not perfectly horizontally symmetric, notably because the king and queen begin on different files and because of castling. In practice, the saved memory and improved statistical sharing can outweigh that lost distinction.

The transformation must be applied consistently to the king square and to every feature square for that perspective. As with any king-bucket change, a king move requires an accumulator refresh; a king move that crosses the mirror axis also changes the canonical orientation of all feature squares.

## Advanced topics: Multiple layers

### The naive cost of dense layers

Consider the apparently modest architecture `A[768] -> N` twice, then `2N -> 8 -> 1`. The two input accumulators require `3N` units to apply SCReLU. A straightforward 16-bit implementation of the `2N -> 8` layer performs one multiply and one accumulation for each of its `16N` connections, costing another `16N` units in this model. Thus the total cost is roughly `20N`, and even worse if `8` becomes `16`.

This is deliberately a naive cost model. The reference SCReLU produces 32-bit values; practical deeper networks first requantize their bounded activations to 8 bits and use packed byte-dot-product instructions, rather than feeding those 32-bit values directly to a 16-bit dense layer.

For comparison, the direct `2N -> 1` output in the reference architecture costs `3N` for the two SCReLU activations and `4N` for its 32-bit dot product: roughly `7N` dense work. While these estimates do not reflect real CPU inference costs (especially after manual SIMD optimizations), they show why extra layers are not so simple in practice.

Making extra layers viable requires deliberately chosen quantized representations, SIMD-friendly dot products, and small layer shapes. The following sections will build those techniques up before using deeper output networks.

We will focus on the bottleneck affine layer; the remainder of the network is relatively standard, though one must take care to track how quantization changes each layer.

### 8-bit-to-32-bit affine layers

The basic representation for a deeper NNUE affine layer uses unsigned 8-bit activations, signed 8-bit weights, and signed 32-bit biases and outputs. For an `N -> M` layer, it computes:

```text
y[j] = bias[j] + sum_i(x[i] * weight[j][i])
```

where `x[i]` is `uint8`, `weight[j][i]` is `int8`, and `y[j]` is `int32`. The 32-bit accumulator is necessary because each output sums many signed 8-bit products; the next bounded activation rescales and narrows the result back to 8 bits.

This representation is efficient on modern CPUs because SIMD byte-dot-product instructions multiply many 8-bit activation/weight pairs and accumulate their results directly into 32-bit lanes. On x86, `VPDPBUSD` (commonly called `DPBUSD`) accumulates four `uint8 × int8` products into each `int32` lane. On Arm, `USDOT` provides the corresponding unsigned-by-signed byte dot product.

`USDOT` is not available on many older Arm targets, which instead provide signed-byte `SDOT`. To use `SDOT` safely for activations along with `USDOT` and `DPBUSD`, Stockfish requires all `x[i]` values to be in the range `[0, 127]` so that they can safely be interpreted as either `int8` or `uint8`.

Using these instructions, an `N -> M` layer is best optimized when `M` is a multiple of 16. Hardware without specific `DPBUSD` or `(U)SDOT` instructions can still fall back to standard vector multiply-add instructions.

### Pairwise-multiplication activation

An ordinary activation preserves one output value for every input value. The first dense layer after a wide feature transformer would therefore still receive a very wide vector. Pairwise multiplication reduces that width while retaining a useful nonlinearity.

Mathematically, start with an even-length pre-activation vector `a` of width `2H`. Split it into two halves, apply the CReLU defined in Basic theory, and multiply corresponding values elementwise:

```text
z[i] = CReLU(a[i]) * CReLU(a[i + H]) / Q,    i = 0 .. H - 1
```

In a common 8-bit implementation, the activation values are already quantized and CReLU is realized as `clamp(a, 0, 255)`, representing the real-valued `[0, 1]` range at scale 255. The divisor `Q = 512` is a fixed-point rescaling constant, not CReLU's mathematical upper bound, and is the most common choice.

The result has width `H`, so the following affine layer has half as many input dimensions. The two halves are independently learned, making this a simple multiplicative gate.

Pairwise multiplication is closely related to SCReLU. SCReLU squares one clipped value; pairwise multiplication takes the product of two independently learned clipped values. If both inputs were the same, the operation would be identical to SCReLU (modulo quantization).

For readers familiar with machine learning, it also resembles a multiplicative activation or gated unit: one bounded activation modulates another. Unlike common GLU variants, however, both sides use a clipped integer activation rather than a sigmoid gate. For a deeper comparison with gated activations in machine learning, see [Chess networks use a Gated Nonlinear Unit](https://asteri.sm/files/2026-01-13-activation).

Pairwise multiplication can also make the vector substantially sparser. A product is nonzero only when both clipped inputs are nonzero, so `z` has no more nonzeros than either half individually. The increased sparsity allows an effective sparse inference to further reduce the cost of the next layer. Pairing is performed within each perspective's feature-transformer output; the two resulting perspective vectors are still ordered by side to move before entering the next layer.

#### Stockfish implementation note

See [Stockfish's in-code explanation](https://github.com/official-stockfish/Stockfish/blob/48a9118251609a0902df7889f7fe241921ac358f/src/nnue/nnue_feature_transformer.h#L239-L280) for the current best known SIMD implementation of pairwise multiplication.

### Sparse inference

After pairwise multiplication, many 8-bit activations are zero. For a Stockfish-sized feature transformer with width 1024, around 60% zeros is a reasonable representative figure, though the exact rate depends on the network and positions. A normal affine layer would still multiply every input by every output weight. Sparse inference instead records the indices of nonzero input blocks and computes only their contributions:

```text
y = bias
for each nonzero four-value block b:
    y += dot(x[4b : 4b + 4], weight_block[b])
```

#### Four-value blocks

The block size is four activations because `DPBUSD` and `USDOT` each form their 32-bit partial sums from four byte products. It is therefore only worthwhile to test and skip an entire four-value block. If even one value in the block is nonzero, its dot-product instruction must still load all four activation bytes and all four corresponding weights; testing individual values would add bookkeeping without avoiding that instruction.

The weights are laid out by input block, with the `M` output weights needed for one block contiguous. A SIMD kernel can then use `DPBUSD` or `USDOT` to add that block's contribution to an entire output tile at once. With `K` nonzero blocks, the useful work is proportional to `K * M`, rather than `(N / 4) * M` packed byte-dot-product operations for a dense pass.

#### Finding nonzero blocks

The block indices should be produced with SIMD as well. Treat each four-byte block as a 32-bit lane and compare many lanes with zero in one vector operation. The most optimized way to store and query these indices may vary depending on the instruction set.

#### Offline permutation for block sparsity

Block sparsity depends on which four activations are adjacent. After training, collect activation statistics and choose an offline permutation that groups channels whose zero patterns tend to coincide. This concentrates zeros into all-zero four-value blocks, increasing the chance that the sparse kernel can skip a `DPBUSD` or `USDOT` operation.

The permutation does not change the network function. Permute the activation channels and apply the inverse permutation to the corresponding columns of the following affine layer (and any producer ordering that must match them). It is a one-time model-layout transformation, not a learned routing decision or per-position work.

Sparse inference helps only when the input is sparse enough to repay its indexing and less-regular memory access. It is most valuable for the first dense layer after the wide, pairwise-reduced feature-transformer output. Later layers are narrow and may be relatively dense, so ordinary packed affine kernels are usually the better choice.

### Manual unrolling

Depending on the instruction set, manually unrolling the affine loop can improve throughput. Repeatedly updating one accumulator creates a dependency chain: the next instruction must wait for the previous partial sum.

Unroll the loop over input blocks and keep several independent partial sums instead:

```text
acc0 = bias;  acc1 = 0;  acc2 = 0
for each group of three input blocks:
    acc0 = dot_accumulate(acc0, block0, weights0)
    acc1 = dot_accumulate(acc1, block1, weights1)
    acc2 = dot_accumulate(acc2, block2, weights2)
output = acc0 + acc1 + acc2
```

The same idea applies to sparse inference: unroll over several nonzero blocks and merge their partial sums after the loop.

## Advanced topics: Feature set extensions

### Merged kings and HalfKAv2_hm

Within a fixed king bucket, the two king piece categories can be merged without losing information. Let the friendly king be on square `S`. Its king feature can only be “friendly king on `S`”, while the opponent king can occupy any square except `S`. Those two sets of legal features are disjoint, so one shared 64-square king category can encode both: index `S` means the friendly king and every other index means the opponent king.

This reduces the ordinary 12 `(piece type, color)` categories to 11 categories for each king bucket, shrinking the piece-square part of the feature transformer from `12 × 64` to `11 × 64` columns. It is a lossless representation change for legal positions, not a learned approximation.

Stockfish's `HalfKAv2` applies this merged-king encoding. `HalfKAv2_hm` then applies the horizontal mirroring described above, canonicalizing each perspective so its friendly king uses one half of the board. The combination preserves the useful king-relative information while reducing the number of canonical king buckets and the feature-transformer storage.

### Full Threats feature set

Full Threat inputs model a subset of piece pair features, specifically the ones where the first (non-king) piece attacks the second (non-king) piece. For example, "white queen on d1 attacks black knight on h5" is encoded as "us queen on d1 attacks them knight on h5" for White's perspective and "them queen on d8 attacks us knight on h4" for Black's perspective. Threat input features are also mirrored horizontally (where the reflection would apply to both pieces' squares in the pair).

Threat features alone are not sufficient for evaluation (how can you distinguish positions where there are no threats, for instance?) and hence function as an add-on to a base piece feature set such as HalfKAv2_hm. One can think of piece features as providing the foundation, while the threat features add many minor corrections.

#### Deduplicating features

Several threat features are redundant. For example, if a rook is attacking a queen, it is guaranteed that the queen will also be attacking the rook, and hence it is unnecessary to consider any feature of the form "rook attacks queen". Generalizing, we can remove all features where the attacking piece's attack set is a subset of the attacked piece's attack set (pawn to bishop, pawn to queen, bishop to queen, rook to queen).

If the two pieces are identical type (and not pawns), then one feature of the pair will also be redundant. In this case, we remove the feature where the numerical square (0 - 63) of the attacking piece is less than that of the attacked piece. A half-exclusion also applies for pawns of opposite colors attacking each other (but not for pawns of same color, because the symmetry breaks there).

#### I8 quantization for threat feature weights

The number of active and changing threat features depends on the position, but is typically higher when more pieces (especially the queens) are on the board. In a typical midgame position, there might be 3-4x as many changing threat features as piece features, and memory bandwidth becomes a bottleneck for accumulation speed. We thus store threat features as i8 and convert them to i16 on the fly during accumulation. This process seems to increase speed much more on ARM architectures (+10%) compared to x86 (+5%). Because threat feature weights are typically not high in absolute value (see our earlier comment regarding the separation between threat and piece features), we can clip them during training with negligible loss in evaluation quality.

### Pawn-pair features

Pawn-pair features add a binary feature for an unordered pair of pawns in a local three-file band: the same file or either adjacent file. Both pawn squares are restricted to ranks 2 through 7, and each pawn is represented by its square and its color relative to the current perspective. The usual perspective orientation and horizontal mirroring are applied before indexing the pair.

This directly represents local pawn-structure relationships: connected pawns, doubled pawns, opposing pawn contacts, and nearby pawn chains. A three-file neighborhood keeps the feature set focused on relations that are geometrically local while avoiding the cost of all possible pawn pairs. The current encoding has 96 possible pawn identities (2 colors × 48 squares) and therefore `96 choose 2 = 4560` possible unordered pair features. It intentionally includes unused features to allow vectorized indexing via SIMD.

Unlike ordinary piece-square features, a pawn move or capture can change several pair features involving nearby pawns. Pawns move relatively infrequently, so this extra update work can be a reasonable tradeoff for a richer pawn-structure representation.

Pawn pair features also subsume pawn to pawn threat relations in Full_Threats.

Currently, Stockfish stores a combined accumulator for all features. Depending on the engine, separating accumulators by feature type may be superior.

## Miscellaneous

### Deriving dual perspective from symmetry

Dual perspective can also be derived from a symmetry argument. Start with a White-relative feature transformer of width `2N`. Let `R(P)` be the position obtained by vertically reflecting position `P` and exchanging the piece colors. This operation preserves the relative chess position while exchanging the two sides.

For the feature transformer to have the same relative interpretation after this transformation, there must be a permutation `Pi` of its outputs such that:

```text
acc(R(P)) = Pi(acc(P))
```

Applying the reflection twice returns the original position, so `Pi` must be an involution:

```text
Pi(Pi(x)) = x
```

Every output channel of `Pi` is therefore either a fixed point or part of a pair. In terms of feature-transformer parameters, the symmetry requires:

```text
W[R(feature)] = Pi(W[feature])
b = Pi(b)
```

A fixed point is one neuron whose weights already have the required vertical symmetry. This is effectively the ordinary PSQT case: a vertically symmetric table needs no separate dual-perspective accumulator.

For a dual-perspective NNUE, choose `Pi` to exchange two `N`-wide blocks and give it no fixed points:

```text
Pi(acc_white, acc_black) = (acc_black, acc_white)
```

The first block for a position is then the second block for its reflected, color-swapped position. Label the blocks as the White and Black perspectives, and this is exactly the usual dual-perspective feature transformer. Ordering the activated blocks as `stm`, then `nstm`, gives the output layer its consistent relative interpretation.
