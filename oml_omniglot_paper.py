import os.path as osp
import argparse
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import configs.classification.class_parser as class_parser
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification

logger = logging.getLogger("experiment")


def main():
    p = class_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    if args["log_root"]:
        log_root = osp.join("./results", args["log_root"]) + "/"
    else:
        log_root = osp.join("./results/")

    my_experiment = experiment(
        args["name"],
        args,
        log_root,
        commit_changes=False,
        rank=0,
        seed=args["seed"],
    )
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger("experiment")

    # Using first 963 classes of the omniglot as the meta-training set
    # args["classes"] = list(range(963))
    args["classes"] = list(range(args["num_classes"]))
    print("Using classes:", args["num_classes"])
    # logger.info("Using classes:", str(args["num_classes"]))

    # args["traj_classes"] = list(range(int(963 / 2), 963))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False
    dataset_spt = df.DatasetFactory.get_dataset(
        args["dataset"],
        background=True,
        train=True,
        path=args["path"],
        # all=True,
        # all=False,
        all=args["all"],
        prefetch_gpu=args["prefetch_gpu"],
        device=device,
        resize=args["resize"],
        augment=args["augment_spt"],
    )
    dataset_qry = df.DatasetFactory.get_dataset(
        args["dataset"],
        background=True,
        train=True,
        path=args["path"],
        # all=True,
        # all=False,
        all=args["all"],
        prefetch_gpu=args["prefetch_gpu"],
        device=device,
        resize=args["resize"],
        augment=args["augment_qry"],
    )
    dataset_test = df.DatasetFactory.get_dataset(
        args["dataset"],
        background=True,
        train=False,
        path=args["path"],
        # all=True,
        # all=False,
        all=args["all"],
        resize=args["resize"],
        # augment=args["augment"],
    )

    logger.info(f"Support size: {len(dataset_spt)}, Query size: {len(dataset_qry)}, test size: {len(dataset_test)}")
    # print(f"Support size: {len(dataset_spt)}, Query size: {len(dataset_qry)}, test size: {len(dataset_test)}")

    pin_memory = use_cuda
    if args["prefetch_gpu"]:
        num_workers = 0
        pin_memory = False
    else:
        num_workers = args["num_workers"]
    # Iterators used for evaluation
    iterator_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        # pin_memory=pin_memory,
    )

    iterator_train = torch.utils.data.DataLoader(
        dataset_spt,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        # pin_memory=pin_memory,
    )

    logger.info("Support sampler:")
    sampler_spt = ts.SamplerFactory.get_sampler(
        args["dataset"],
        args["classes"],
        dataset_spt,
        dataset_test,
        prefetch_gpu=args["prefetch_gpu"],
        use_cuda=use_cuda,
        num_workers=0,
    )
    logger.info("Query sampler:")
    sampler_qry = ts.SamplerFactory.get_sampler(
        args["dataset"],
        args["classes"],
        dataset_qry,
        dataset_test,
        prefetch_gpu=args["prefetch_gpu"],
        use_cuda=use_cuda,
        num_workers=0,
    )

    config = mf.ModelFactory.get_model(
        "na",
        args["dataset"],
        output_dimension=1000,
        resize=args["resize"],
    )

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu_to_use))
        logger.info("Using gpu : %s", "cuda:" + str(gpu_to_use))
    else:
        device = torch.device("cpu")

    maml = MetaLearingClassification(args, config).to(device)

    for step in range(args["steps"]):

        t1 = np.random.choice(args["classes"], args["tasks"], replace=False)

        d_traj_iterators_spt = []
        d_traj_iterators_qry = []
        for t in t1:
            d_traj_iterators_spt.append(sampler_spt.sample_task([t]))
            d_traj_iterators_qry.append(sampler_qry.sample_task([t]))


        d_rand_iterator = sampler_spt.get_complete_iterator()

        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data_paper(
            d_traj_iterators_spt,
            d_traj_iterators_qry,
            d_rand_iterator,
            steps=args["update_step"],
            reset=not args["no_reset"],
        )
        if torch.cuda.is_available():
            x_spt, y_spt, x_qry, y_qry = (
                x_spt.to(device),
                y_spt.to(device),
                x_qry.to(device),
                y_qry.to(device),
            )

        #
        accs, loss = maml(x_spt, y_spt, x_qry, y_qry)

        # Evaluation during training for sanity checks
        if step % 40 == 5:
            writer.add_scalar("/metatrain/train/accuracy", accs[-1], step)
            writer.add_scalar("/metatrain/train/loss", loss[-1], step)
            writer.add_scalar("/metatrain/train/accuracy0", accs[0], step)
            writer.add_scalar("/metatrain/train/loss0", loss[0], step)
            logger.info("step: %d \t training acc %s", step, str(accs))
            logger.info("step: %d \t training loss %s", step, str(loss))
        # Currently useless
        if (step % 300 == 3) or ((step + 1) == args["steps"]):
            torch.save(maml.net, my_experiment.path + "learner.model")
        #     utils.log_accuracy(maml, my_experiment, iterator_test, device, writer, step)
        #     utils.log_accuracy(
        #         maml, my_experiment, iterator_train, device, writer, step
        #     )


if __name__ == "__main__":
    main()
