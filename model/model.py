import torch
from torch import nn

from .config import ModelConfig
from .modules import LayerStacks, get_feature_cls
from .quantize import QuantizationManager


class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_name: str,
        config: ModelConfig,
        num_psqt_buckets: int = 8,
        num_ls_buckets: int = 8,
    ):
        super().__init__()

        feature_cls = get_feature_cls(feature_name)
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.quantize_config = config.quantize_config
        self.quantization = QuantizationManager(config.quantize_config)

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        self.input = feature_cls(self.L1 + self.num_psqt_buckets)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config, self.quantization)

        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.init_weights(num_psqt_buckets, self.quantization.nnue2score)

        self.gumbel_tau = config.gumbel_tau
        self.num_router_features_per_side = config.num_router_features_per_side
        self.router = nn.Linear(self.num_router_features_per_side * 2, self.num_ls_buckets)
        self.router_ls = nn.Parameter(1.0 * torch.ones(1))
        self.logits_probe = nn.Identity()

    @torch.no_grad()
    def clip_weights(self):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        for group in self.weight_clipping:
            for p in group["params"]:
                if "min_weight" in group or "max_weight" in group:
                    p_data_fp32 = p.data
                    min_weight = group["min_weight"]
                    max_weight = group["max_weight"]
                    if "virtual_params" in group:
                        virtual_params = group["virtual_params"]
                        xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                        ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                        expanded_virtual_layer = virtual_params.repeat(xs, ys)
                        if min_weight is not None:
                            min_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                                - expanded_virtual_layer
                            )
                        if max_weight is not None:
                            max_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                                - expanded_virtual_layer
                            )
                    p_data_fp32.clamp_(min_weight, max_weight)

    def clip_input_weights(self):
        self.input.clip_weights(self.quantization)

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        white_values: torch.Tensor,
        black_indices: torch.Tensor,
        black_values: torch.Tensor,
        psqt_indices: torch.Tensor,
        layer_stack_indices: torch.Tensor,
    ):
        _, _ = psqt_indices, layer_stack_indices # legacy compatibility, no longer used
        wp, bp = self.input(white_indices, white_values, black_indices, black_values)
        w, wpsqt = torch.split(wp, self.L1, dim=1)
        b, bpsqt = torch.split(bp, self.L1, dim=1)
        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        l0_ = torch.clamp(l0_, 0.0, self.quantization.max_ft_activation)

        l0_s = torch.split(l0_, self.L1 // 2, dim=1)
        l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]

        router_features = torch.cat(
            [l0_s1[0][:, -self.num_router_features_per_side:], l0_s1[1][:, -self.num_router_features_per_side:]],
            dim=1,
        )

        l0_ = torch.cat(l0_s1, dim=1) * self.quantization.l0_correction_factor

        routing_logits = self.router_ls * self.router(router_features)
        # Gumbel-Softmax with hard=True produces a one-hot tensor with attached gradients.
        routing_weights = torch.nn.functional.gumbel_softmax(routing_logits, tau=self.gumbel_tau, hard=True)
        # Pass both the raw logits and the hard routing weights to the probe
        self.logits_probe((routing_logits, routing_weights))

        if self.training:
            x = self.layer_stacks(l0_, routing_weights)

            # Apply STE multiplication for PSQT
            wpsqt_reshaped = wpsqt.view(-1, self.num_psqt_buckets, 1)
            bpsqt_reshaped = bpsqt.view(-1, self.num_psqt_buckets, 1)

            wpsqt_selected = (wpsqt_reshaped * routing_weights.unsqueeze(-1)).sum(dim=1)
            bpsqt_selected = (bpsqt_reshaped * routing_weights.unsqueeze(-1)).sum(dim=1)
        else:
            dynamic_indices = routing_logits.argmax(dim=1)
            x = self.layer_stacks(l0_, dynamic_indices)

            psqt_indices_unsq = dynamic_indices.unsqueeze(dim=1)
            wpsqt_selected = wpsqt.gather(1, psqt_indices_unsq)
            bpsqt_selected = bpsqt.gather(1, psqt_indices_unsq)

        x = x + (wpsqt_selected - bpsqt_selected) * (us - 0.5)

        return x
