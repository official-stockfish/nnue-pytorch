import argparse


def make_action(class_name, field_name):
    class SetConfig(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(class_name, field_name, values)

    return SetConfig


# 3 layer fully connected network
class ModelConfig:
    L1: int = 1024
    L2: int = 31
    L3: int = 32

    @staticmethod
    def add_model_args(parser):
        parser.add_argument(
            "--l1",
            type=int,
            default=ModelConfig.L1,
            action=make_action(ModelConfig, "L1"),
        )
        parser.add_argument(
            "--l2",
            type=int,
            default=ModelConfig.L2,
            action=make_action(ModelConfig, "L2"),
        )


# parameters needed for the definition of the loss
class LossParams:
    in_offset: float = 270
    out_offset: float = 270
    in_scaling: float = 340
    out_scaling: float = 380
    start_lambda: float = 1.0
    end_lambda: float = 1.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0
    w1: float = 0.0
    w2: float = 0.5

    @staticmethod
    def add_loss_args(parser):
        parser.add_argument(
            "--in-offset",
            type=float,
            default=LossParams.in_offset,
            action=make_action(LossParams, "in_offset"),
        )
        parser.add_argument(
            "--out-offset",
            type=float,
            default=LossParams.out_offset,
            action=make_action(LossParams, "out_offset"),
        )
        parser.add_argument(
            "--in-scaling",
            type=float,
            default=LossParams.in_scaling,
            action=make_action(LossParams, "in_scaling"),
        )
        parser.add_argument(
            "--out-scaling",
            type=float,
            default=LossParams.out_scaling,
            action=make_action(LossParams, "out_scaling"),
        )
        parser.add_argument(
            "--start-lambda",
            type=float,
            default=LossParams.start_lambda,
            action=make_action(LossParams, "start_lambda"),
        )
        parser.add_argument(
            "--end-lambda",
            type=float,
            default=LossParams.end_lambda,
            action=make_action(LossParams, "end_lambda"),
        )
        parser.add_argument(
            "--pow-exp",
            type=float,
            default=LossParams.pow_exp,
            action=make_action(LossParams, "pow_exp"),
        )
        parser.add_argument(
            "--qp-asymmetry",
            type=float,
            default=LossParams.qp_asymmetry,
            action=make_action(LossParams, "qp_asymmetry"),
        )
        parser.add_argument(
            "--w1",
            type=float,
            default=LossParams.w1,
            action=make_action(LossParams, "w1"),
        )
        parser.add_argument(
            "--w2",
            type=float,
            default=LossParams.w2,
            action=make_action(LossParams, "w2"),
        )
