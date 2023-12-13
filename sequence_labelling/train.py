import argparse
import logging
import pickle
from pathlib import Path

import torch
import transformers
from load import load_data
from model import LangModelWithDense
from torch.utils.tensorboard import SummaryWriter
from torchcrf import CRF
from tqdm import tqdm
from utils import Meter, dump_args, init_logger, print_info

logger = logging.getLogger("bert.train")


def train_model(
    args,
    model,
    train_loader,
    dev_loader,
    optimizer,
    criterion,
    num_classes,
    target_classes,
    label_encoder,
    device,
):
    # create to Meter's classes to track the performance of the model during
    # training and evaluating
    train_meter = Meter(target_classes)
    dev_meter = Meter(target_classes)

    run_name = f"-{args.run_name}" if args.run_name else ""
    tb_writer = SummaryWriter(comment=run_name)

    best_f1 = -1

    # print the metrics
    desc_fmt = "Train Loss: {:.4f}, Train Micro F1: {:.4f}, Train Macro F1: {:.4f}"
    # epoch loop
    for epoch in range(args.epochs):
        print()
        print(f"Epoch {epoch+1}/{args.epochs}")

        model.train()

        train_tqdm = tqdm(train_loader)
        # train loop
        for train_x, train_y, mask, crf_mask in train_tqdm:
            # get the logits and update the gradients
            optimizer.zero_grad()

            logits = model.forward(train_x, mask)

            if args.crf:
                loss = -criterion(
                    logits.to(device), train_y, reduction="token_mean", mask=crf_mask
                )
            else:
                loss = criterion(
                    logits.reshape(-1, num_classes).to(device),
                    train_y.reshape(-1).to(device),
                )

            loss.backward()
            optimizer.step()

            # get the current metrics (average over all the train)
            loss, _, _, micro_f1, _, _, macro_f1 = train_meter.update_params(
                loss.item(), logits, train_y
            )

            train_tqdm.set_description(desc_fmt.format(loss, micro_f1, macro_f1))
            train_tqdm.refresh()

        tb_writer.add_scalar("Train/loss", train_meter.loss, epoch)
        tb_writer.add_scalar("Train/macro_f1", train_meter.macro_f1, epoch)
        tb_writer.flush()

        logger.info(
            "[{}/{}] Train Loss: {:.4f}, Train Macro F1: {:.4f}".format(
                epoch + 1, args.epochs, train_meter.loss, train_meter.macro_f1
            )
        )

        train_meter.reset()

        model.eval()

        # evaluation loop -> mostly same as the training loop, but without
        # updating the parameters
        macro_f1 = 0
        dev_tqdm = tqdm(dev_loader)
        for dev_x, dev_y, mask, crf_mask in dev_tqdm:
            logits = model.forward(dev_x, mask)

            if args.crf:
                loss = -criterion(
                    logits.to(device), dev_y, reduction="token_mean", mask=crf_mask
                )
            else:
                loss = criterion(
                    logits.reshape(-1, num_classes).to(device),
                    dev_y.reshape(-1).to(device),
                )

            loss, _, _, micro_f1, _, _, macro_f1 = dev_meter.update_params(
                loss.item(), logits, dev_y
            )

            dev_tqdm.set_description(
                "Dev Loss: {:.4f}, Dev Micro F1: {:.4f}, Dev Macro F1: {:.4f}".format(
                    loss, micro_f1, macro_f1
                )
            )
            dev_tqdm.refresh()

        tb_writer.add_scalar("Dev/loss", dev_meter.loss, epoch)
        tb_writer.add_scalar("Dev/macro_f1", dev_meter.macro_f1, epoch)
        tb_writer.flush()

        logger.info(
            "[{}/{}] Dev Loss: {:.4f}, Dev Macro F1: {:.4f}".format(
                epoch + 1, args.epochs, dev_meter.loss, dev_meter.macro_f1
            )
        )

        dev_meter.reset()

        # if the current macro F1 score is the best one -> save the model
        if macro_f1 > best_f1:
            args.save_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Macro F1 score improved from {:.4f} -> {:.4f}. Saving model...".format(
                    best_f1, macro_f1
                )
            )

            best_f1 = macro_f1
            torch.save(model, args.save_path / "model.pt")
            with (args.save_path / "label_encoder.pk").open("wb") as file:
                pickle.dump(label_encoder, file)
            with (args.save_path / "best").open("w") as file:
                file.write(f"epoch: {epoch + 1} macro_f1: {best_f1}")

    tb_writer.close()


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # Loading the train and dev data and save them in a loader + the encoder of
    # the classes
    logger.info("Loading data")
    train_loader, dev_loader, label_encoder = load_data(
        args.train_path,
        args.dev_path,
        args.batch_size,
        args.tokens_column,
        args.predict_column,
        args.lang_model_name,
        args.max_len,
        args.separator,
        args.pad_label,
        args.null_label,
        device,
    )
    logger.info("Loaded data")

    # select the desired language model and get the embeddings size
    lang_model = transformers.AutoModel.from_pretrained(args.lang_model_name)
    input_size = 768 if "base" in args.lang_model_name else 1024

    # create the model, the optimizer (weights are set to 0 for <pad> and <X>)
    # and the loss function
    model = LangModelWithDense(
        lang_model, input_size, len(label_encoder.classes_), args.fine_tune
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.crf:
        criterion = CRF(len(label_encoder.classes_), batch_first=True).to(device)
    else:
        weights = torch.tensor(
            [
                1 if label not in [args.pad_label, args.null_label] else 0
                for label in label_encoder.classes_
            ],
            dtype=torch.float32,
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # remove the null_label (<X>) and the pad label (<pad>) from the evaluated
    # targets during training
    classes = label_encoder.classes_.tolist()  # type: ignore
    classes.remove(args.null_label)
    classes.remove(args.pad_label)
    target_classes = [
        label_encoder.transform([clss])[0] for clss in classes  # type: ignore
    ]

    print_info(
        target_classes, label_encoder, args.lang_model_name, args.fine_tune, device
    )

    # start training
    train_model(
        args,
        model,
        train_loader,
        dev_loader,
        optimizer,
        criterion,
        len(label_encoder.classes_),
        target_classes,
        label_encoder,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str, help="Path to the training file")
    parser.add_argument("dev_path", type=str, help="Path to the dev file")
    parser.add_argument("tokens_column", type=int, help="The column of the tokens.")
    parser.add_argument(
        "predict_column", type=int, help="The column that must be predicted"
    )
    parser.add_argument(
        "lang_model_name",
        type=str,
        help="Language model name of HuggingFace's implementation.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument(
        "--save_path", type=Path, default="models", help="Where to save the model/"
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Use this to fine-tune the language model's weights.",
    )
    parser.add_argument(
        "--max_len", type=int, default=192, help="Maximum length of the files."
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=" ",
        help="Separator of the tokens in the train/dev files.",
    )
    parser.add_argument("--pad_label", type=str, default="<pad>", help="The pad token.")
    parser.add_argument("--null_label", type=str, default="<X>", help="The null token.")
    parser.add_argument(
        "--crf",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use this to remove the CRF on top of the language model.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="The device to train on."
    )
    parser.add_argument(
        "--run-name", type=str, help="Suffix for the Tensorboard run name"
    )
    parser.add_argument(
        "--logfile",
        type=str,
        help="Path to the destination log file",
        default="train.log",
    )
    parser.add_argument(
        "--lr", type=float, help="Optimiser learning rate", default=2e-4
    )
    parser.add_argument(
        "--log-all",
        action="store_true",
        help="Enable logging of everything, including libraries like transformers",
    )

    args = parser.parse_args()

    log_name = None if args.log_all else "bert"
    init_logger(args.logfile, log_name)
    dump_args(args)
    main(args)
