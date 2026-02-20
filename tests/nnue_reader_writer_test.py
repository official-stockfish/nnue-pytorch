import sys
import os
import io
import tempfile
import torch
import numpy as np

# Dynamically add the project root to sys.path so 'model' can be imported
# This allows you to run `python tests/test_serialize.py` directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import model as M
from model.utils.serialize import NNUEWriter, NNUEReader, encode_leb_128_array, decode_leb_128_array
from model.model import NNUEModel
from model.config import ModelConfig
from model.features import FeatureSet
from model.quantize import QuantizationConfig


def run_serialization_roundtrip(feature_name: str, is_full_threats: bool):
    print(f"--- Testing Serialization Roundtrip for: {feature_name} ---")
    
    # 1. Initialize actual configurations
    config = ModelConfig()
        
    feature_set = M.get_feature_set_from_name(feature_name)
    
    quant_config = QuantizationConfig()
    
    # 2. Initialize the actual model
    model = NNUEModel(feature_set, config, quant_config)
    
    with torch.no_grad():
        model.input.weight.data.fill_(0.05)
        model.input.bias.data.fill_(0.5)
        for stack in [model.layer_stacks.l1, model.layer_stacks.l2, model.layer_stacks.output]:
            stack.linear.weight.data.fill_(0.05)
            stack.linear.bias.data.fill_(0.75)

    original_ft_weight = model.input.weight.data.clone()
    original_fc_weight = model.layer_stacks.l1.linear.weight.data.clone()

    # 3. Write to the writer buffer
    description = f"Real instance test for {feature_name}"
    writer = NNUEWriter(model, description=description)
    
    # --- THE FIX: Use a real temporary file instead of io.BytesIO ---
    fd, temp_path = tempfile.mkstemp(suffix=".nnue")
    
    try:
        # Write the byte array to the physical temp file
        with os.fdopen(fd, 'wb') as f:
            f.write(writer.buf)
            
        # 4. Read back from the physical file
        with open(temp_path, 'rb') as f:
            reader = NNUEReader(f, feature_set, config, quant_config)

        # 5. Assertions
        assert reader.description == description, f"Description mismatch! Expected '{description}', got '{reader.description}'"
        
        tolerance = 1/8

        read_ft_weight = reader.model.input.weight.data
        ft_match = torch.allclose(original_ft_weight, read_ft_weight, atol=tolerance)
        assert ft_match, "Feature Transformer weights failed to roundtrip correctly."

        read_fc_weight = reader.model.layer_stacks.l1.linear.weight.data
        fc_match = torch.allclose(original_fc_weight, read_fc_weight, atol=tolerance)
        assert fc_match, "Fully Connected weights failed to roundtrip correctly."

        print(f"Successfully verified read/write protocol for {feature_name}!\n")

    finally:
        # Clean up the temporary file from the disk
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_leb128_logic():
    print("--- Testing LEB128 Encoding/Decoding ---")
    # Testing bounds and negative numbers to ensure bit-shifting is accurate
    original = np.array([0, 64, 127, 128, -1, -128, 16384, -16384], dtype=np.int32)
    encoded = encode_leb_128_array(original)
    decoded = decode_leb_128_array(bytes(encoded), len(original))
    
    assert np.array_equal(original, decoded), f"LEB128 mismatch: {original} vs {decoded}"
    print("LEB128 logic passed!\n")


if __name__ == "__main__":
    print("Starting NNUE Serialization Tests...\n")
    try:
        # Test 1: LEB128 array compression utilities
        test_leb128_logic()
        run_serialization_roundtrip(feature_name="Full_Threats", is_full_threats=True)
        run_serialization_roundtrip(feature_name="HalfKAv2", is_full_threats=False)
        
        print("ALL TESTS PASSED SUCCESSFULLY.")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)