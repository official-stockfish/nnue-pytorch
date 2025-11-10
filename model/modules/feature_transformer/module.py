import math

import torch
from torch import nn

from .functions import (
    FeatureTransformerSliceFunction,
    DoubleFeatureTransformerSliceFunction,
)


class BaseFeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        sigma = math.sqrt(1 / num_inputs)

        self.weight = nn.Parameter(
            torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma)
            - sigma
        )
        self.bias = nn.Parameter(
            torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma
        )


class FeatureTransformerSlice(BaseFeatureTransformerSlice):
    def forward(self, feature_indices, feature_values):
        return FeatureTransformerSliceFunction.apply(
            feature_indices, feature_values, self.weight, self.bias
        )


class DoubleFeatureTransformerSlice(BaseFeatureTransformerSlice):
    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        return DoubleFeatureTransformerSliceFunction.apply(
            feature_indices_0,
            feature_values_0,
            feature_indices_1,
            feature_values_1,
            self.weight,
            self.bias,
        )


if __name__ == "__main__":
    import time

    def FeatureTransformerSliceFunctionEmulate(
        feature_indices, feature_values, weight, bias
    ):
        batch_size = feature_indices.shape[0]
        num_inputs = weight.shape[0]
        max_active_features = feature_indices.shape[1]
        inputs = torch.zeros(
            batch_size, num_inputs, dtype=torch.float32, device=weight.device
        )
        for i in range(batch_size):
            for j in range(max_active_features):
                feature = feature_indices[i, j]
                value = feature_values[i, j]
                inputs[i, feature] += value

        return torch.mm(inputs, weight) + bias

    def test():
        BATCH_SIZE = 16
        INPUT_SIZE = 10
        MAX_ACTIVE_FEATURES = 32
        STRIDE = 128
        MAX_ERROR = 1e-4

        torch.manual_seed(0)
        weight0 = torch.rand(
            INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True
        )
        bias0 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        torch.manual_seed(0)
        weight1 = torch.rand(
            INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True
        )
        bias1 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        indices0 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(
            dtype=torch.int32
        )
        indices1 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(
            dtype=torch.int32
        )
        values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)
        values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)

        output00 = FeatureTransformerSliceFunctionEmulate(
            indices0.clone(), values0.clone(), weight0, bias0
        )
        output01 = FeatureTransformerSliceFunctionEmulate(
            indices1.clone(), values1.clone(), weight0, bias0
        )
        # output10 = FeatureTransformerSliceFunction.apply(indices0.clone().cuda(), values0.clone().cuda(), weight1.cuda(), bias1.cuda())
        # output11 = FeatureTransformerSliceFunction.apply(indices1.clone().cuda(), values1.clone().cuda(), weight1.cuda(), bias1.cuda())
        output10, output11 = DoubleFeatureTransformerSliceFunction.apply(
            indices0.clone().cuda(),
            values0.clone().cuda(),
            indices1.clone().cuda(),
            values1.clone().cuda(),
            weight1.cuda(),
            bias1.cuda(),
        )

        assert torch.max(output00.cpu() - output10.cpu()) < MAX_ERROR
        assert torch.max(output01.cpu() - output11.cpu()) < MAX_ERROR
        (output00 - output01).sum().backward()
        (output10 - output11).sum().backward()
        assert torch.max(weight0.grad.cpu() - weight1.grad.cpu()) < MAX_ERROR
        assert torch.max(bias0.grad.cpu() - bias1.grad.cpu()) < MAX_ERROR
        print("Tests passed.")

    def bench():
        INPUT_SIZE = 40960
        BATCH_SIZE = 8192
        ITERS = 64
        STRIDE = 264
        MAX_ACTIVE_FEATURES = 64

        layer = DoubleFeatureTransformerSlice(INPUT_SIZE, STRIDE).cuda()
        indices0 = torch.cat(
            [
                torch.sort(
                    (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4) * INPUT_SIZE),
                    dim=1,
                )[0].to(dtype=torch.int32),
                torch.full(
                    (BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32
                ),
            ],
            dim=1,
        ).cuda()
        values0 = torch.rand(
            BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32
        ).cuda()
        indices1 = torch.cat(
            [
                torch.sort(
                    (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4)) * INPUT_SIZE,
                    dim=1,
                )[0].to(dtype=torch.int32),
                torch.full(
                    (BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32
                ),
            ],
            dim=1,
        ).cuda()
        values1 = torch.rand(
            BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32
        ).cuda()

        output0, output1 = layer(indices0, values0, indices1, values1)

        start = time.time()

        for _ in range(ITERS):
            output0, output1 = layer(indices0, values0, indices1, values1)
            output0 = torch.clamp(output0, 0.0, 1.0)
            output1 = torch.clamp(output1, 0.0, 1.0)

            g = ((output0 - output1) ** 2).mean()
            g.backward()

            torch.cuda.synchronize()

        end = time.time()

        # for param in layer.parameters():
        #    print(param.grad)

        print("{} pos/s".format((ITERS * BATCH_SIZE) / (end - start)))

    test()
    bench()
