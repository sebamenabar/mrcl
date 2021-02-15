import os
import os.path as osp
import sys

from argparse import ArgumentParser
import logging
import json

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import configs.classification.class_parser_eval as class_parser_eval
import datasets.datasetfactory as df
import model.learner as Learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment

from easydict import EasyDict as edict


def load_model(args, config):
    # if args["model_path"] is not None:
    if args.model_path is not None:
        net_old = Learner.Learner(config)
        # logger.info("Loading model from path %s", args["model_path"])
        # net = torch.load(args["model_path"], map_location="cpu")
        net = torch.load(args.model_path, map_location="cpu")

        for (n1, old_model), (n2, loaded_model) in zip(
            net_old.named_parameters(), net.named_parameters()
        ):
            # print(n1, n2, old_model.adaptation, old_model.meta)
            loaded_model.adaptation = old_model.adaptation
            loaded_model.meta = old_model.meta

        net.reset_vars()
    else:
        net = Learner.Learner(config)
    return net


def eval_iterator(iterator, device, maml):
    correct = 0
    for img, target in iterator:
        img = img.to(device)
        target = target.to(device)
        logits_q = maml(img)

        pred_q = (logits_q).argmax(dim=1)

        correct += torch.eq(pred_q, target).sum().item() / len(img)
    return correct / len(iterator)


def train_iterator(iterator_sorted, device, maml, opt):
    for img, y in iterator_sorted:
        img = img.to(device)
        y = y.to(device)

        pred = maml(img)
        opt.zero_grad()
        loss = F.cross_entropy(pred, y)
        loss.backward()
        opt.step()


def eval_loop(
    # args,
    trainset,
    valset,
    learner,
    optimizer,
    runs,
    reset_zero,
    device,
    lr_sweep_range,
    prefix="",
):

    print(f"Eval loop {optimizer} reset zero {reset_zero}")

    final_results_train = []
    final_results_test = []
    lr_sweep_results = []

    # args['schedule'] = [int(x) for x in args['schedule'].split(":")]
    # args.schedule = [10, 25, 50, 75, 100, 200, 300]
    schedule = [25, 100, 200, 300]
    # no_of_classes_schedule = args["schedule"]
    no_of_classes_schedule = schedule
    # print(args.schedule)
    for total_classes in schedule:

        print(f"\n\n--------  Beggining schedule {total_classes} ---------- \n ")

        # lr_sweep_range = [0.03, 0.01, 0.003,0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        lr_all = []
        for lr_search_runs in range(0, 5):

            classes_to_keep = np.random.choice(
                list(range(664, 964)), total_classes, replace=False
            )

            dataset = utils.remove_classes_omni(trainset, classes_to_keep)

            iterator_sorted = torch.utils.data.DataLoader(
                utils.iterator_sorter_omni(
                    dataset,
                    False,
                    # classes=no_of_classes_schedule,
                ),
                batch_size=1,
                # shuffle=args["iid"],
                shuffle=False,
                num_workers=1,
            )

            dataset = utils.remove_classes_omni(trainset, classes_to_keep)
            iterator_train = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=False, num_workers=1
            )

            max_acc = -1000
            # lr_sweep_range = [
            #     # 0.03,
            #     # 0.01,
            #     # 0.003,
            #     # 0.001,
            #     0.0003,
            #     # 0.0001,
            #     # 0.00003,
            #     # 0.00001,
            #     # 5e-6,
            #     # 1e-6,
            # ]
            for lr in lr_sweep_range:

                learner.reset_vars(zero=reset_zero)

                if optimizer == "adam":
                    opt = torch.optim.Adam(learner.get_adaptation_parameters(), lr=lr)
                elif optimizer == "sgd":
                    opt = torch.optim.SGD(
                        learner.get_adaptation_parameters(),
                        lr=lr,
                        weight_decay=5e-4,
                        momentum=0.9,
                    )

                train_iterator(iterator_sorted, device, learner, opt)

                correct = eval_iterator(iterator_train, device, learner)
                if correct > max_acc:
                    max_acc = correct
                    max_lr = lr

                print(f"Accuracy LR {lr}: {correct}")

            lr_all.append(max_lr)
            results_mem_size = (max_acc, max_lr)
            lr_sweep_results.append((total_classes, results_mem_size))

            # my_experiment.results["LR Search Results"] = lr_sweep_results
            # my_experiment.store_json()
            # logger.debug("LR RESULTS = %s", str(lr_sweep_results))
        # print("SCHEDULE %d RESULTS = %s" % (total_classes, str(lr_sweep_results)))

        from scipy import stats

        best_lr = float(stats.mode(lr_all)[0][0])

        # logger.info("BEST LR %s= ", str(best_lr))
        print("BEST LR =%s " % str(best_lr))

        for current_run in range(0, runs):

            # classes_to_keep = np.random.choice(list(range(650)), total_classes, replace=False)
            classes_to_keep = np.random.choice(
                list(range(664, 964)), total_classes, replace=False
            )

            dataset = utils.remove_classes_omni(trainset, classes_to_keep)

            iterator_sorted = torch.utils.data.DataLoader(
                utils.iterator_sorter_omni(
                    dataset, False, classes=no_of_classes_schedule
                ),
                batch_size=1,
                # shuffle=args["iid"],
                shuffle=False,
                num_workers=2,
            )

            dataset = utils.remove_classes_omni(valset, classes_to_keep)
            iterator_test = torch.utils.data.DataLoader(
                dataset, batch_size=32, shuffle=False, num_workers=1
            )

            dataset = utils.remove_classes_omni(trainset, classes_to_keep)
            iterator_train = torch.utils.data.DataLoader(
                dataset, batch_size=32, shuffle=False, num_workers=1
            )

            lr = best_lr

            learner.reset_vars(zero=reset_zero)

            if optimizer == "adam":
                opt = torch.optim.Adam(learner.get_adaptation_parameters(), lr=lr)
            elif optimizer == "sgd":
                opt = torch.optim.SGD(
                    learner.get_adaptation_parameters(),
                    lr=lr,
                    weight_decay=5e-4,
                    momentum=0.9,
                )

            train_iterator(iterator_sorted, device, learner, opt)

            # logger.info("Result after one epoch for LR = %f", lr)
            print("Result after one epoch for LR = %f" % lr)

            correct = eval_iterator(iterator_train, device, learner)

            correct_test = eval_iterator(iterator_test, device, learner)

            results_mem_size = (correct, best_lr, "train")
            # logger.info("Final Max Result train = %s", str(correct))
            print("Final Max Result train = %s" % str(correct))
            final_results_train.append((total_classes, results_mem_size))

            results_mem_size = (correct_test, best_lr, "test")
            # logger.info("Final Max Result test= %s", str(correct_test))
            print("Final Max Result test= %s" % str(correct_test))
            final_results_test.append((total_classes, results_mem_size))

            # my_experiment.results["Final Results"] = final_results_train
            # my_experiment.results["Final Results Test"] = final_results_test
            # my_experiment.store_json()
            # logger.debug("FINAL RESULTS = %s", str(final_results_train))
            # print("FINAL RESULTS = %s" % str(final_results_train))
            # logger.debug("FINAL RESULTS = %s", str(final_results_test))
            # print("FINAL RESULTS = %s" % str(final_results_test))
    print("Final results train")
    print(final_results_train)
    print("Final results test")
    print(final_results_test)

    with open(
        osp.join(args.exp_dir, f"{prefix}cl_results_{optimizer}_zero_{reset_zero}.json"), "w"
    ) as f:
        json.dump(
            {
                "final_results_train": final_results_train,
                "final_results_test": final_results_test,
            },
            f,
        )


def run_eval(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device", device)
    else:
        device = torch.device("cpu")

    args.model_path = osp.join(args.exp_dir, args.model_name)

    with open(osp.join(args.exp_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    resize = metadata["params"]["resize"]

    config = mf.ModelFactory.get_model(
        # "na", args["dataset"], output_dimension=1000, resize=resize
        "na", args.dataset, output_dimension=1000, resize=resize
    )

    maml = load_model(args, config)
    maml = maml.to(device)

    reset_zero = False
    maml.reset_vars(zero=reset_zero)

    data_train = df.DatasetFactory.get_dataset(
        args.dataset, train=True, background=True, path=args.data_path, resize=resize
    )
    data_test = df.DatasetFactory.get_dataset(
        args.dataset, train=False, background=True, path=args.data_path, resize=resize
    )

    to_keep = np.arange(664, 964)
    trainset = utils.remove_classes_omni(data_train, to_keep)
    valset = utils.remove_classes_omni(data_test, to_keep)

    print(len(trainset), len(valset))

    # for optimizer in ["sgd", "adam"]:
    for optimizer in ["adam"]: # adam worked best
        for reset_zero in [True, False]:
            eval_loop(
                trainset,
                valset,
                maml,
                optimizer,
                10,
                reset_zero,
                device,
                args.lr_sweep_range,
                args.prefix,
            )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="../")
    parser.add_argument("--dataset", default="omniglot")
    parser.add_argument("--exp_dir")
    parser.add_argument("--model_name")
    parser.add_argument("--lr_sweep_range", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    parser.add_argument("--prefix", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)