import argparse

import numpy as np
from numpy.lib.twodim_base import tril_indices
import torch
from torch.nn import functional as F
from tqdm import tqdm

import datasets.datasetfactory as df
import configs.classification.pretraining_parser as params
import model.learner as learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment
import logging

logger = logging.getLogger("experiment")


def main():
    p = params.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    my_experiment = experiment(
        args["name"], args, "./results/", commit_changes=False, rank=0, seed=1
    )

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu_to_use))
        logger.info("Using gpu : %s", "cuda:" + str(gpu_to_use))
    else:
        device = torch.device("cpu")

    print("Train dataset")
    dataset = df.DatasetFactory.get_dataset(
        args["dataset"],
        background=True,
        train=True,
        path=args["path"],
        all=True,
        resize=args["resize"],
        augment=args["augment"],
    )
    print("Val dataset")
    val_dataset = df.DatasetFactory.get_dataset(
        args["dataset"],
        background=True,
        train=True,
        path=args["path"],
        all=True,
        resize=args["resize"],
        # augment=args["augment"],
    )

    train_labels = np.arange(664)
    class_labels = np.array(dataset.targets)
    labels_mapping = {
        tl: (class_labels == tl).astype(int).nonzero()[0] for tl in train_labels
    }
    train_indices = [tl[:15] for tl in labels_mapping.values()]
    val_indices = [tl[15:] for tl in labels_mapping.values()]
    train_indices = [i for sublist in train_indices for i in sublist]
    val_indices = [i for sublist in val_indices for i in sublist]

    # indices = np.zeros_like(class_labels)
    # for a in train_labels:
    #     indices = indices + (class_labels == a).astype(int)
    # val_indices = (indices == 0).astype(int)
    # indices = np.nonzero(indices)[0]
    trainset = torch.utils.data.Subset(dataset, train_indices)

    # print(indices)
    print("Total samples:", len(class_labels))
    print("Train samples:", len(train_indices))
    print("Val samples:", len(val_indices))

    #  val_labels = np.arange(664)
    # class_labels = np.array(dataset.targets)
    # val_indices = np.zeros_like(class_labels)
    # for a in train_labels:
    #     val_indices = val_indices + (class_labels != a).astype(int)
    # val_indices = np.nonzero(val_indices)[0]
    valset = torch.utils.data.Subset(val_dataset, val_indices)

    train_iterator = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_iterator = torch.utils.data.DataLoader(
        valset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    logger.info(str(args))

    config = mf.ModelFactory.get_model("na", args["dataset"], resize=args["resize"])

    maml = learner.Learner(config).to(device)

    for k, v in maml.named_parameters():
        print(k, v.requires_grad)

    # opt = torch.optim.Adam(maml.parameters(), lr=args["lr"])
    opt = torch.optim.SGD(
        maml.parameters(),
        lr=args["lr"],
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=[200, 250],
        gamma=0.1,
    )

    best_val_acc = 0

    # print(learner)
    # print(learner.eval(False))

    for e in range(args["epoch"]):
        correct = 0
        total_loss = 0
        # learner.train()
        for img, y in tqdm(train_iterator):
            img = img.to(device)
            y = y.to(device)
            pred = maml(img)

            opt.zero_grad()
            loss = F.cross_entropy(pred, y.long())
            loss.backward()
            opt.step()
            correct += (pred.argmax(1) == y).sum().float() / len(y)
            total_loss += loss.item()
        scheduler.step()

        val_correct = 0
        val_total_loss = 0
        # learner.eval()
        for img, y in tqdm(val_iterator):
            img = img.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = maml(img)

                opt.zero_grad()
                loss = F.cross_entropy(pred, y.long())
                # loss.backward()
                # opt.step()
                val_correct += (pred.argmax(1) == y).sum().float().item()
                val_total_loss += loss.item() * y.size(0)

        val_acc = val_correct / len(val_indices)
        val_loss = val_total_loss / len(val_indices)

        logger.info("Accuracy at epoch %d = %s", e, str(correct / len(train_iterator)))
        logger.info("Loss at epoch %d = %s", e, str(total_loss / len(train_iterator)))
        logger.info("Val Accuracy at epoch %d = %s", e, str(val_acc))
        logger.info("Val Loss at epoch %d = %s", e, str(val_loss))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"\nNew best validation accuracy: {str(best_val_acc)}\n")
            torch.save(maml, my_experiment.path + "model_best.net")

    torch.save(maml, my_experiment.path + "last_model.net")


if __name__ == "__main__":

    main()
