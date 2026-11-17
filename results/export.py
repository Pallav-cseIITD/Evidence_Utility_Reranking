"""Export compact tables from the two reproduced experiment summaries."""

import argparse
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(hotpot, qampari, output):
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.mkdir(parents=True)
    for name, value in (("hotpot_results.json", hotpot), ("qampari_results.json", qampari)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_csv(output / "hotpot_evidence.csv", [dict(method=name, **values)
        for name, values in hotpot["evidence"]["methods"].items()
        if name not in ("fusion_reverse", "repair_reverse")])
    answers = [dict(reader="qwen8b", diagnostic_only=False, method=name, questions=values["questions"],
                    answer_em=values["answer_em"], attributed_em=values["attributed_em"], invalid=values["invalid"])
               for name, values in hotpot["primary"]["metrics"].items()]
    for role, reader in (("A", "qwen14b"), ("B", "phi4")):
        for name, values in hotpot["additional"][role]["methods"].items():
            answers.append(dict(reader=reader, diagnostic_only=True, method=name,
                questions=values["completed"], answer_em=values["answer_em"],
                attributed_em=values["attributed_em"], invalid=values["invalid"]))
    write_csv(output / "hotpot_answers.csv", answers)
    write_csv(output / "qampari_reader_b.csv", [dict(method=name, questions=value["questions"], valid=value["valid"],
        **value["metrics"]) for name, value in qampari["results"]["B"].items()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hotpot", type=Path)
    parser.add_argument("qampari", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(json.loads(args.hotpot.read_text(encoding="utf-8")),
           json.loads(args.qampari.read_text(encoding="utf-8")), args.output)
