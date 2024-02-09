#!/usr/bin/env python

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import typer


def parse_instance(answer: str) -> dict[str, list[str]]:
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return {"Cause": [], "Effect": []}

    causes, _, effects = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())

    return {"Cause": causes, "Effect": effects}


def parse(row: Mapping[str, str], col: str) -> tuple[str | None, str | None]:
    d = parse_instance(row[col])
    if not d["Cause"] or not d["Effect"]:
        return None, None
    return d["Cause"][0], d["Effect"][0]


def clean_str(s: str) -> str:
    s = s.casefold().strip()
    return re.sub(r"\s", "", s)


def symm_substr(a: str, b: str) -> bool:
    a = clean_str(a)
    b = clean_str(b)
    return a in b or b in a


def excess_words(a: str, b: str) -> str:
    a = a.casefold().strip()
    b = b.casefold().strip()

    if a in b:
        return b.replace(a, "")
    else:
        return a.replace(b, "")


def excess_words_count(a: str, b: str) -> int:
    return len(excess_words(a, b).split())


def main(infile: Path, outfile: Path) -> None:
    df = pd.read_json(infile)

    df[["pred_cause", "pred_effect"]] = df.apply(
        parse, col="output", axis="columns", result_type="expand"
    )
    df[["gold_cause", "gold_effect"]] = df.apply(
        parse, col="gold", axis="columns", result_type="expand"
    )
    df = df.dropna()

    df["cause_substr"] = df.apply(
        lambda x: symm_substr(x["pred_cause"], x["gold_cause"]), axis="columns"
    )
    df["effect_substr"] = df.apply(
        lambda x: symm_substr(x["pred_effect"], x["gold_effect"]), axis="columns"
    )

    df_diff = df.query("pred_cause != gold_cause or pred_effect != gold_effect")
    df_substr = df_diff.query("cause_substr and effect_substr").copy()

    df_substr["cause_excess_count"] = df_substr.apply(
        lambda x: excess_words_count(x["pred_cause"], x["gold_cause"]), axis="columns"
    )
    df_substr["effect_excess_count"] = df_substr.apply(
        lambda x: excess_words_count(x["pred_effect"], x["gold_effect"]), axis="columns"
    )

    print(f"{len(df_substr)=}")
    df_substr.to_json(outfile, orient="records")


if __name__ == "__main__":
    typer.run(main)
