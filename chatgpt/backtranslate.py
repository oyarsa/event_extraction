import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import typer
from google_trans_new import google_translator
from tqdm import tqdm


def dbg(*args: Any, **kwargs: Any) -> None:
    i: int = kwargs.pop("i", 0)
    n: int | None = kwargs.pop("n", None)
    if n is not None and i < n:
        print(*args, **kwargs)


def translate(
    translator: google_translator,
    sentences: list[str],
    src: str,
    dest: str,
    desc: str,
    max_n_print: int = 10,
) -> list[str]:
    translated: list[str] = []
    for i, s in enumerate(tqdm(sentences, desc=f"Translating {desc}")):
        dbg("source: ", s, i=i, n=max_n_print)
        t = translator.translate(s, lang_src=src, lang_tgt=dest)
        dbg("translated: ", t, i=i, n=max_n_print)
        dbg(i=i, n=max_n_print)
        translated.append(t)
    return translated
    # return [translator.translate(s, lang_src=src, lang_tgt=dest) for s in sentences]


def backtranslate(data: list[dict[str, str]], *, key: str, lang: str) -> list[str]:
    translator = google_translator()
    sentences = [d[key] for d in data]
    translated_pt = translate(
        translator, sentences, src="en", dest=lang, desc=f"en->{lang}"
    )
    translated_en = translate(
        translator, translated_pt, src=lang, dest="en", desc=f"{lang}->en"
    )
    return translated_en


def main(
    data_path: Path, output_path: Path, key: str = "sentence2", lang: str = "fr"
) -> None:
    "en->fr->en seems to be the best. en->pt->en generates very similar test."
    data = json.loads(data_path.read_text())
    aug = deepcopy(data)

    augmented_sentences = backtranslate(data, key=key, lang=lang)
    for i, d in enumerate(aug):
        d[f"{key}_og"] = d[key]
        d[key] = augmented_sentences[i]

    output_path.write_text(json.dumps(aug))


if __name__ == "__main__":
    typer.run(main)
