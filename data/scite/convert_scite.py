"""Convert SCITE's XML data to the tagged JSON format.

Obtained from https://github.com/Das-Boot/scite under the Apache-2.0 license.

Li, Zhaoning and Li, Qi and Zou, Xiaotian and Ren, Jiangtao
http://www.sciencedirect.com/science/article/pii/S0925231220316027
Neurocomputing
pp. 207-219
Causality extraction based on self-attentive BiLSTM-CRF with transferred embeddings
Volume 423
2021

NB: Some entries in the dataset are incorrect, such as missing slashes in the XML tags,
or incorrectly encoded characters. The script will ignore these entries.
"""

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import TextIO
from xml.etree import ElementTree


@dataclass(frozen=True)
class ParsedInstance:
    cause: str
    effect: str
    sentence: str


def parse_events(label: str, sentence: str) -> list[ParsedInstance]:
    # sourcery skip: identity-comprehension
    """Parse the label and extract the events.

    parse_events(
        label="Cause-Effect((e2,e1),(e1,e3))"
        sentence="The <e1>influence</e1> of <e2>family</e2> on <e3>adolescent behavior</e3>."
    )
    >>> [
        {"cause": "family", "effect": "influence", "sentence": ...sentence..},
        {"cause": "influence", "effect": "adolescent behavior", "sentence": ...sentence..},
    ]

    The output sentence has the event tags removed.
    """
    pattern_pair = r"\((\w+),(\w+)\)"
    event_pairs: list[tuple[str, str]] = re.findall(pattern_pair, label)

    pattern_text = r"<(\w+)>([^<]+)<\/?\1>"
    event_texts: dict[str, str] = {
        event_id: event_text
        for event_id, event_text in re.findall(pattern_text, sentence)
    }

    clean_sentence = re.sub(r"</?e\d+>", "", sentence).strip()

    parsed_events: list[ParsedInstance] = []
    for cause_id, effect_id in event_pairs:
        try:
            parsed_events.append(
                ParsedInstance(
                    cause=event_texts[cause_id],
                    effect=event_texts[effect_id],
                    sentence=clean_sentence,
                )
            )
        except KeyError:
            continue

    return parsed_events


def parse_data(input_file: TextIO, events_per_sentence: int) -> list[ParsedInstance]:
    """Process the XML file and extract causal sentences."""
    root = ElementTree.parse(input_file).getroot()
    parsed_events: list[ParsedInstance] = []

    for item in root.findall("item"):
        label = item.get("label")
        if not (label and "Cause-Effect" in label):
            continue

        sentence_tag = item.find("sentence")
        if sentence_tag is None:
            continue

        sentence = sentence_tag.text
        if sentence is None:
            continue

        if events := parse_events(label, sentence):
            parsed_events.extend(events[:events_per_sentence])

    return parsed_events


def hash_instance(data: list[str], length: int = 8) -> str:
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()[:length]


@dataclass(frozen=True)
class TaggedInstance:
    context: str
    answers: str
    question: str
    question_type: str
    id: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "id", hash_instance([self.context, self.answers]))


def rewrite_tag(events: list[ParsedInstance]) -> list[TaggedInstance]:
    """Rewrite the events data using the tagged format."""
    question = "What are the events?"
    question_type = "cause"
    template = "[Cause] {cause} [Relation] cause [Effect] {effect}"

    return [
        TaggedInstance(
            context=event.sentence,
            question=question,
            question_type=question_type,
            answers=template.format(cause=event.cause, effect=event.effect),
        )
        for event in events
    ]


def main(input_file: TextIO, output_file: TextIO, events_per_sentence: int):
    """Main function to process the XML file and save the output as JSON."""
    parsed_data = parse_data(input_file, events_per_sentence)
    tagged_data = rewrite_tag(parsed_data)

    json.dump([asdict(d) for d in tagged_data], output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XML to JSON")
    parser.add_argument(
        "input_file", type=argparse.FileType("r"), help="Input XML file"
    )
    parser.add_argument(
        "output_file", type=argparse.FileType("w"), help="Output JSON file"
    )
    parser.add_argument(
        "--events-per-sentence",
        type=int,
        default=1,
        help="Number of events to extract per sentence",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.events_per_sentence)
