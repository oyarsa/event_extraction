import json
from typing import Any

import annotate
import classifier
import simple_parsing


def separate(
    data: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    "Separates data into rule-based classes (EM+no-substr) and substrings."
    done: list[dict[str, Any]] = []
    todo: list[dict[str, Any]] = []

    for d in data:
        valid = annotate.annotate_entry(d["gold"], d["output"])
        if valid is not None:
            done.append({**d, "valid": valid})
        else:
            todo.append(d)

    return done, todo


def main() -> None:
    config = simple_parsing.parse(classifier.Config, add_config_path_arg=True)
    classifier.set_seed(config.seed)
    classifier.suppress_transformers_warnings()

    tokenizer, model = classifier.init_model(config.model_name)
    device = classifier.get_device()

    assert config.test_data_path is not None
    data = json.loads(config.test_data_path.read_text())[: config.max_samples]
    done, todo = separate(data)

    loader, _ = classifier.preprocess_data(
        todo,
        tokenizer,
        config.max_seq_length,
        1,
        config.batch_size,
        config.use_passage,
        has_labels=False,
    )

    results = classifier.predict(model, loader, device, "eval")
    todo_results = [
        {
            "valid": bool(results.preds[i]),
            "input": results.passages[i],
            "output": results.outputs[i],
            "gold": results.annotations[i],
        }
        for i in range(len(results.preds))
    ]
    final = done + todo_results

    valid = sum(d["valid"] for d in final)
    substr_valid = sum(d["valid"] for d in todo_results)
    substr_invalid = sum(not d["valid"] for d in todo_results)
    em = sum(d["valid"] for d in done)
    nosubstr = sum(not d["valid"] for d in done)

    print()
    print(f"Total: {len(data)}")
    print(f"Valid: {valid}/{len(final)} ({valid / len(final):.2%})")
    print(f"EM + no-substr: {len(done)}")
    print(f"Substr: {len(todo)}")

    print()
    print(
        f"Substr valid: {substr_valid}/{len(todo_results)} ({substr_valid / len(todo_results):.2%})"
    )
    print(
        f"Substr invalid: {substr_invalid}/{len(todo_results)} ({substr_invalid / len(todo_results):.2%})"
    )
    print(f"EM: {em}/{len(done)} ({em / len(done):.2%})")
    print(f"No substr: {nosubstr}/{len(done)} ({nosubstr / len(done):.2%})")


if __name__ == "__main__":
    main()
