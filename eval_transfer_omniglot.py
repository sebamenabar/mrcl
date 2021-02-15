import os
import os.path as osp
import sys

from argparse import ArgumentParser
import logging
import json
import copy

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


def eval_loop(
    # args,
    train_iterator,
    val_iterator,
    maml,
    optimizer,
    runs,
    reset_zero,
    device,
    prefix,
):

    # opt = None
    # reset_zero = False
    # optimizer = "sgd"

    # train_iterator = torch.utils.data.DataLoader(
    #     data_train, batch_size=64, shuffle=True, num_workers=0, drop_last=True
    # )
    # val_iterator = torch.utils.data.DataLoader(
    #     data_test, batch_size=64, shuffle=True, num_workers=0, drop_last=False
    # )

    # lr_all = []

    lr_sweep_range = [
        # 1e-1,
        # 5e-2,
        3e-2,
        1e-2,
        5e-3,
        1e-3,
        5e-4,
        # 1e-4,
        ]
    lr_all = {lr: [] for lr in lr_sweep_range}
    for lr_search_runs in range(runs):
        # max_acc = -1000
        for lr in lr_sweep_range:

            maml_copy = copy.deepcopy(maml)
            maml_copy.load_state_dict(maml.state_dict())
            maml_copy = maml_copy.to(device)

            if reset_zero:
                torch.nn.init.zeros_(maml_copy.parameters()[-2])
            else:
                torch.nn.init.kaiming_normal_(maml_copy.parameters()[-2])
            torch.nn.init.zeros_(maml_copy.parameters()[-1])

            if optimizer == "sgd":
                opt = torch.optim.SGD(
                    filter(lambda x: x.requires_grad, maml_copy.parameters()),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=5e-4,
                )
            elif optimizer == "adam":
                opt = torch.optim.Adam(
                    filter(lambda x: x.requires_grad, maml_copy.parameters()),
                    lr=lr,
                )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[40], gamma=0.1)

            epochs = 40
            best_val_acc = 0
            best_val_epoch = 0
            for e in range(epochs):
                correct = 0
                total_loss = 0
                # learner.train()
                for img, y in (train_iterator):
                    img = img.to(device)
                    y = y.to(device)
                    pred = maml_copy(img)

                    opt.zero_grad()
                    loss = F.cross_entropy(pred, y.long())
                    loss.backward()
                    opt.step()
                    correct += (pred.argmax(1) == y).sum().float() / len(y)
                    total_loss += loss.item()
                scheduler.step()

                val_correct = 0
                val_total_loss = 0
                total_val_samples = 0
                # learner.eval()
                for img, y in (val_iterator):
                    img = img.to(device)
                    y = y.to(device)
                    with torch.no_grad():
                        pred = maml_copy(img)

                        opt.zero_grad()
                        loss = F.cross_entropy(pred, y.long())
                        # loss.backward()
                        # opt.step()
                        val_correct += (pred.argmax(1) == y).sum().float().item()
                        val_total_loss += loss.item() * y.size(0)
                        total_val_samples += y.size(0)

                val_acc = val_correct / total_val_samples
                val_loss = val_total_loss / total_val_samples
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = e

            # print("Accuracy at epoch %d = %s" % (e, str(correct / len(train_iterator))))
            # print("Loss at epoch %d = %s" % (e, str(total_loss / len(train_iterator))))
            print(f"Best val Accuracy lr {lr} at epoch {best_val_epoch} = {best_val_acc}")
            # print("Val Loss at epoch %d = %s" % (e, str(val_loss)))
            lr_all[lr].append((best_val_acc, best_val_epoch))


    print("Final results train")
    print(lr_all)
    # print("Final results test")
    # print(final_results_test)

    with open(
        osp.join(args.exp_dir, f"{prefix}_transfer_results_{optimizer}_zero_{reset_zero}.json"), "w"
    ) as f:
        json.dump(
            {
                # "final_results_train": final_results_train,
                "lr_all": lr_all,
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

    train_iterator = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=0, drop_last=True
    )
    val_iterator = torch.utils.data.DataLoader(
        valset, batch_size=64, shuffle=True, num_workers=0, drop_last=False
    )

    for param in maml.parameters():
        param.requires_grad = False
    maml.parameters()[-2].requires_grad = True
    maml.parameters()[-1].requires_grad = True

    print("Features only")
    for optimizer in ["sgd", "adam"]:
    # for optimizer in ["adam"]: # adam worked best
        print(optimizer)
        # for reset_zero in [True, False]:
        for reset_zero in [True]:
            print(reset_zero)
            eval_loop(
                train_iterator,
                val_iterator,
                maml,
                optimizer,
                4,
                reset_zero,
                device,
                "features",
            )

    for param in maml.parameters():
        param.requires_grad = True
    # maml.parameters()[-2].requires_grad = True
    # maml.parameters()[-1].requires_grad = True

    print("full finetune")
    # for optimizer in ["sgd", "adam"]:
    for optimizer in ["adam"]: # adam worker best for full finetune
        print(optimizer)
        # for reset_zero in [True, False]:
        for reset_zero in [True]: # with zero worked
            print(reset_zero)
            eval_loop(
                train_iterator,
                val_iterator,
                maml,
                optimizer,
                4,
                reset_zero,
                device,
                "full_finetune",
            )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="../")
    parser.add_argument("--dataset", default="omniglot")
    parser.add_argument("--exp_dir")
    parser.add_argument("--model_name")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)