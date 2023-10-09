#!/usr/bin/env python

import re
import sys

import pandas as pd


def parse_instance(answer: str) -> tuple[dict[str, list[str]], str | None]:
    """Parse string answer to separate into class and spans
    Simple case:
    [Cause] This is a cause [Effect] This is an effect

    Complex case:
    [Cause] This cause 1 | This cause 2 [Effect] This effect 1 | This effect 2
    """
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return {
            "Cause": [],
            "Effect": [],
        }, "cause"
    causes, relation, effects = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())
    relation = relation.strip()

    return {
        "Cause": causes,
        "Effect": effects,
    }, relation


def parse(row: dict[str, str], col: str) -> tuple[str | None, str | None]:
    d, _ = parse_instance(row[col])
    if not d["Cause"] or not d["Effect"]:
        return None, None
    return d["Cause"][0], d["Effect"][0]


def clean_str(s: str) -> str:
    s = s.lower().strip()
    return re.sub(r"\s", "", s)


def symm_substr(a: str, b: str) -> bool:
    a = clean_str(a)
    b = clean_str(b)
    return a in b or b in a


def excess_words(a: str, b: str) -> str:
    a = a.lower().strip()
    b = b.lower().strip()

    if a in b:
        return b.replace(a, "")
    else:
        return a.replace(b, "")


def excess_words_count(a: str, b: str) -> int:
    return len(excess_words(a, b).split())


def main() -> None:
    infile = sys.argv[1]
    outname = sys.argv[2]
    outfile_substr = outname + "_substr.json"
    outfile_nosubstr = outname + "_nosubstr.json"

    data = pd.read_json(infile)
    print("Parsed output:", parse_instance(data.iloc[0].output))
    df = data.copy()
    df[["pred_cause", "pred_effect"]] = df.apply(
        parse, col="output", axis=1, result_type="expand"
    )
    df[["gold_cause", "gold_effect"]] = df.apply(
        parse, col="gold", axis=1, result_type="expand"
    )
    df = df.dropna()

    print(len(df.query("pred_effect != gold_effect")))
    df["cause_substr"] = df.apply(
        lambda x: symm_substr(x["pred_cause"], x["gold_cause"]), axis=1
    )
    print(
        "Number of instances where the predicted cause is a substring of the gold cause:",
        df.query("pred_cause != gold_cause")["cause_substr"].value_counts(),
    )

    df["effect_substr"] = df.apply(
        lambda x: symm_substr(x["pred_effect"], x["gold_effect"]), axis=1
    )
    print(
        "Number of instances where the predicted effect is a substring of the gold effect:",
        df.query("pred_cause != gold_cause")["effect_substr"].value_counts(),
    )

    df_cause = df.query("(pred_cause != gold_cause) and cause_substr").copy()
    df_cause["cause_excess"] = df_cause.apply(
        lambda x: excess_words(x["pred_cause"], x["gold_cause"]), axis=1
    )
    df_cause["cause_excess_count"] = df_cause.apply(
        lambda x: excess_words_count(x["pred_cause"], x["gold_cause"]), axis=1
    )
    print("cause_excess_count", df_cause["cause_excess_count"].describe())

    df_effect = df.query("(pred_effect != gold_effect) and effect_substr").copy()
    df_effect["effect_excess"] = df_effect.apply(
        lambda x: excess_words(x["pred_effect"], x["gold_effect"]), axis=1
    )
    df_effect["effect_excess_count"] = df_effect.apply(
        lambda x: excess_words_count(x["pred_effect"], x["gold_effect"]), axis=1
    )
    print("effect_excess_count", df_effect["effect_excess_count"].describe())

    df_diff = df.query("pred_cause != gold_cause or pred_effect != gold_effect")
    print(
        "Number of instances where the predicted cause or effect is different from the gold cause or effect:",
        len(df_diff),
    )
    print(
        "Number of instances where the predicted cause is the same as the gold cause:",
        df.query("pred_cause == gold_cause")["cause_substr"].value_counts(),
    )

    print("cause_substr", df_diff["cause_substr"].value_counts())
    print("effect_substr", df_diff["effect_substr"].value_counts())
    print(
        "cause_substr and effect_substr",
        (df_diff["cause_substr"] & df_diff["effect_substr"]).value_counts(),
    )

    df_substr = df_diff.query("cause_substr and effect_substr").copy()
    print("cause_substr and effect_substr", len(df_substr))
    df_substr["cause_excess"] = df_substr.apply(
        lambda x: excess_words(x["pred_cause"], x["gold_cause"]), axis=1
    )
    df_substr["cause_excess_count"] = df_substr.apply(
        lambda x: excess_words_count(x["pred_cause"], x["gold_cause"]), axis=1
    )
    print("cause_excess_count", df_substr["cause_excess_count"].describe())

    df_substr["effect_excess"] = df_substr.apply(
        lambda x: excess_words(x["pred_effect"], x["gold_effect"]), axis=1
    )
    df_substr["effect_excess_count"] = df_substr.apply(
        lambda x: excess_words_count(x["pred_effect"], x["gold_effect"]), axis=1
    )
    print("effect_excess_count", df_substr["effect_excess_count"].describe())

    df_substr.to_json(outfile_substr, orient="records")

    df_nosub = df.query(
        "(pred_cause != gold_cause) and (not cause_substr or not effect_substr)"
    ).copy()
    nosub_agg = (
        df_nosub.groupby(["cause_substr", "effect_substr"])["input"]
        .count()
        .reset_index()
    )

    print("nosub_agg")
    print(nosub_agg.to_markdown(tablefmt="simple", index=False))
    print("df_nosub.shape", df_nosub.shape)

    df_nosub.to_json(outfile_nosubstr, orient="records")


if __name__ == "__main__":
    main()
