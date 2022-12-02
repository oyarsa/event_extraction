import argparse
import logging
import os
import pickle

import torch
import transformers
from tqdm import tqdm

from load import load_data_from_file
from utils import dump_args, init_logger

logger = logging.getLogger("bert.predict")


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    os.makedirs(args.output_path, exist_ok=True)

    with open(os.path.join(args.model_path, "label_encoder.pk"), "rb") as file:
        label_encoder = pickle.load(file)

    test_loader, _ = load_data_from_file(
        args.test_path,
        1,
        args.token_column,
        args.predict_column,
        args.lang_model_name,
        512,
        args.separator,
        args.pad_label,
        args.null_label,
        device,
        label_encoder,
        False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.lang_model_name, use_fast=True
    )

    model = torch.load(
        os.path.join(args.model_path, "model.pt"), map_location=args.device
    )
    model.fine_tune = False
    model.eval()

    list_labels = []

    logger.info("Predicting tags")
    for test_x, _, mask, _ in tqdm(test_loader):
        logits = model.forward(test_x, mask)
        preds = torch.argmax(logits, 2)

        end = mask.argmin(1) - 1
        labels = label_encoder.inverse_transform(preds[0][1:end].tolist())
        list_labels.append(labels)

    in_path = args.test_path
    out_path = os.path.join(args.output_path, args.output_name)
    with open(in_path, "r", encoding="utf-8") as in_file, open(
        out_path, "w", encoding="utf-8"
    ) as out_file:
        sentence_idx = 0
        label_idx = 0

        for line in in_file:
            if not line.startswith("#"):
                if line not in [" ", "\n"]:
                    tokens = line.strip().split(args.separator)

                    token = tokens[args.token_column]
                    gold = tokens[args.predict_column]
                    pred = list_labels[sentence_idx][label_idx]

                    out_file.write(args.separator.join([token, gold, pred]) + "\n")

                    subtokens = tokenizer.encode(token, add_special_tokens=False)
                    label_idx += len(subtokens)
                else:
                    assert label_idx == len(list_labels[sentence_idx])

                    out_file.write("\n")
                    sentence_idx += 1
                    label_idx = 0
            else:
                out_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path")
    parser.add_argument("model_path", type=str)
    parser.add_argument("token_column", type=int)
    parser.add_argument("predict_column", type=int)
    parser.add_argument("lang_model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--output_name", type=str, default="predict.conllu")
    parser.add_argument("--separator", type=str, default=" ")
    parser.add_argument("--pad_label", type=str, default="<pad>")
    parser.add_argument("--null_label", type=str, default="<X>")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--log-all",
        action="store_true",
        help="Enable logging of everything, including libraries like transformers",
    )

    args = parser.parse_args()

    log_name = None if args.log_all else "bert"
    init_logger(log_name=log_name)
    dump_args(args)

    main(args)
