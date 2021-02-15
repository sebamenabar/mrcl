import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/regression/empty.ini",
                 help='config file path')
        #
        self.add('--steps', type=int, help='epoch number', default=200000)
        self.add('--gpus', type=int, help='meta-level outer learning rate', default=1)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--tasks', nargs='+', type=int, help='meta batch size, namely task num', default=[1])
        self.add('--meta_lr', nargs='+', type=float, help='meta-level outer learning rate', default=[1e-4])
        self.add('--update_lr', nargs='+', type=float, help='task-level inner update learning rate', default=[0.01])
        self.add('--update_step', nargs='+', type=int, help='task-level inner update steps', default=[10])
        self.add('--dataset', help='Name of experiment', default="omniglot")
        self.add("--no-reset", action="store_true")
        self.add("--reset", default="random", choices=["random", "zero"])
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--path', help='Path of the dataset', default="../")
        self.add('--prefetch_gpu', action="store_true", default=False)
        self.add("--num_classes", default=963, type=int)
        self.add("--num_workers", default=0, type=int)
        self.add("--resize", type=int, default=None)
        self.add("--augment_spt", default=False, action="store_true")
        self.add("--augment_qry", default=False, action="store_true")
        self.add("--log_root", default="")
        self.add("--all", default=False, action="store_true")


