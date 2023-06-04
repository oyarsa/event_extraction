import dataclasses
import re
from importlib import util
from pathlib import Path
from types import ModuleType

import typer


def import_file(full_name: str, path: Path) -> ModuleType:
    "From https://stackoverflow.com/a/46198651"
    spec = util.spec_from_file_location(full_name, path)
    assert spec is not None and spec.loader is not None
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def used_fields(lines: list[str], fields: set[str]) -> tuple[set[str], list[str]]:
    used_fields: set[str] = set()
    used_lines: list[str] = []
    for i, line in enumerate(lines):
        for field in fields:
            if field in used_fields:
                continue
            if re.search(rf"\b{field}\b", line):
                used_lines.append(f"[{i:>3}] {line}")
                used_fields.add(field)
    return used_fields, used_lines


def main(module_file: Path, data_classes: list[str]) -> None:
    mod = import_file(module_file.stem, module_file)
    for data_class in data_classes:
        print(f"Data Class: {data_class}\n")

        fields = {f.name for f in dataclasses.fields(getattr(mod, data_class))}
        mod_lines = module_file.read_text().splitlines()
        main_line = next(i for i, line in enumerate(mod_lines) if "def main(" in line)
        used, lines = used_fields(mod_lines[main_line:], fields)
        unused = fields - used

        print("Used lines:")
        for line in lines:
            print(line)
        print("\nUnused:")
        for field in unused:
            print(f"- {field}")
        print(f"\n{'*'*80}\n")


if __name__ == "__main__":
    typer.run(main)
