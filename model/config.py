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
        )
        parser.add_argument(
            "--l2",
            type=int,
            default=ModelConfig.L2,
        )

    @staticmethod
    def get_model_config(args) -> "ModelConfig":
        config = ModelConfig()
        config.L1 = args.l1
        config.L2 = args.l2
        return config


# parameters needed for the definition of the loss
class LossParams:
    in_offset: float = 270
    out_offset: float = 270
    in_scaling: float = 340
    out_scaling: float = 380
    start_lambda: float | None = None
    end_lambda: float | None = None
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
            dest="in_offset",
            help="offset for conversion to win on input (default=270.0)",
        )
        parser.add_argument(
            "--out-offset",
            type=float,
            default=LossParams.out_offset,
            dest="out_offset",
            help="offset for conversion to win on input (default=270.0)",
        )
        parser.add_argument(
            "--in-scaling",
            type=float,
            default=LossParams.in_scaling,
            dest="in_scaling",
            help="scaling for conversion to win on input (default=340.0)",
        )
        parser.add_argument(
            "--out-scaling",
            type=float,
            default=LossParams.out_scaling,
            dest="out_scaling",
            help="scaling for conversion to win on input (default=380.0)",
        )
        parser.add_argument(
            "--start-lambda",
            type=float,
            default=LossParams.start_lambda,
            dest="start_lambda",
            help="lambda to use at first epoch.",
        )
        parser.add_argument(
            "--end-lambda",
            type=float,
            default=LossParams.end_lambda,
            dest="end_lambda",
            help="lambda to use at last epoch.",
        )
        parser.add_argument(
            "--pow-exp",
            type=float,
            default=LossParams.pow_exp,
            dest="pow_exp",
            help="exponent of the power law used for the mean error (default=2.5)",
        )
        parser.add_argument(
            "--qp-asymmetry",
            type=float,
            default=LossParams.qp_asymmetry,
            dest="qp_asymmetry",
            help="Adjust to loss for those if q (prediction) > p (reference) (default=0.0)",
        )
        parser.add_argument(
            "--w1",
            type=float,
            default=LossParams.w1,
            dest="w1",
            help="weight boost parameter 1 (default=0.0)",
        )
        parser.add_argument(
            "--w2",
            type=float,
            default=LossParams.w2,
            dest="w2",
            help="weight boost parameter 2 (default=0.5)",
        )
        parser.add_argument(
            "--lambda",
            default=1.0,
            type=float,
            dest="lambda_",
            help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).",
        )

    @staticmethod
    def get_loss_params_from_args(args) -> "LossParams":
        params = LossParams()
        params.in_offset = args.in_offset
        params.out_offset = args.out_offset
        params.in_scaling = args.in_scaling
        params.out_scaling = args.out_scaling
        params.start_lambda = args.start_lambda or args.lambda_
        params.end_lambda = args.end_lambda or args.lambda_
        params.pow_exp = args.pow_exp
        params.qp_asymmetry = args.qp_asymmetry
        params.w1 = args.w1
        params.w2 = args.w2
        return params
