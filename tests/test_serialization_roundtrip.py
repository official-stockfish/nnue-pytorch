import io
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_check_eval import init_random_weights
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.utils.serialize import NNUEReader, NNUEWriter


def test_random_weights_roundtrip():
    torch.manual_seed(42)

    config = ModelConfig()
    model = NNUEModel(
        feature_name=DEFAULT_FEATURES,
        config=config,
    )
    # Initialize random weights covering full dynamic range and hitting clipping thresholds
    init_random_weights(model)

    # 1. Export initial model
    writer1 = NNUEWriter(model, description="Test Random Net", verbose=False)
    bytes1 = bytes(writer1.buf)
    assert len(bytes1) > 0

    # 2. Read back into model2
    reader1 = NNUEReader(io.BytesIO(bytes1), DEFAULT_FEATURES, config)
    model2 = reader1.model

    # 3. Export model2
    writer2 = NNUEWriter(model2, description=reader1.description, verbose=False)
    bytes2 = bytes(writer2.buf)

    assert bytes1 == bytes2, (
        f"Round-trip 1 mismatch: {len(bytes1)} vs {len(bytes2)} bytes"
    )

    # 4. Read back into model3
    reader2 = NNUEReader(io.BytesIO(bytes2), DEFAULT_FEATURES, config)
    model3 = reader2.model

    # 5. Export model3
    writer3 = NNUEWriter(model3, description=reader2.description, verbose=False)
    bytes3 = bytes(writer3.buf)

    assert bytes2 == bytes3, (
        f"Round-trip 2 mismatch: {len(bytes2)} vs {len(bytes3)} bytes"
    )

    # Verify model parameters and buffers match between model2 and model3
    for (n2, p2), (n3, p3) in zip(model2.named_parameters(), model3.named_parameters()):
        assert torch.equal(p2, p3), f"Parameter {n2} mismatch between model2 and model3"

    for (n2, b2), (n3, b3) in zip(model2.named_buffers(), model3.named_buffers()):
        assert torch.equal(b2, b3), f"Buffer {n2} mismatch between model2 and model3"
