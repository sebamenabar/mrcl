import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()

        self.add("--gpus", type=int, help="meta-level outer learning rate", default=1)
        self.add("--rank", type=int, help="meta batch size, namely task num", default=0)
        self.add("--seed", nargs="+", help="Seed", default=[90], type=int)
        self.add("--path", help="Path of the dataset", default="../")
        self.add("--epoch", type=int, nargs="+", help="epoch number", default=[120])
        self.add("--dataset", help="Name of experiment", default="omniglot")
        self.add(
            "--lr",
            nargs="+",
            type=float,
            help="task-level inner update learning rate",
            default=[5e-2],
        )
        self.add("--name", help="Name of experiment", default="baseline")
        self.add("--resize", type=int, default=None)
        self.add("--augment", default=False, action="store_true")
        self.add("--prefetch_gpu", action="store_true", default=False)
        self.add("--log_root", default="")
        self.add("--schedule", nargs="+", type=int, default=[60, 90])
