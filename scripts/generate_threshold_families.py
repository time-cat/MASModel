"""Generate a small benchmark suite for identifying budget thresholds.

Each output family has a known distribution of theoretical minimum budgets
``B0``.  The generated files are independent and use deterministic seeds.  A
manifest records the exact generator settings so that model outcomes can be
compared against the known structural distribution.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_FAMILIES = {
    "threshold-2-narrow": {"distribution": "2:1.0", "difficulty": "medium"},
    "threshold-5-narrow": {"distribution": "5:1.0", "difficulty": "medium"},
    "threshold-8-narrow": {"distribution": "8:1.0", "difficulty": "easy"},
    "threshold-8-medium": {"distribution": "8:1.0", "difficulty": "medium"},
    "threshold-8-hard": {"distribution": "8:1.0", "difficulty": "hard"},    
    "threshold-2-5-mixed": {
        "distribution": "2:0.5,5:0.5",
        "difficulty": "medium",
    },
    "threshold-2-5-8-spread": {
        "distribution": "2:0.25,5:0.5,8:0.25",
        "difficulty": "medium",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/threshold_families"))
    parser.add_argument("--num-instances", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--families",
        type=str,
        default=None,
        help="Optional name=distribution pairs separated by ';'.",
    )
    parser.add_argument(
        "--structure-distribution",
        default="chain:0.34,wide:0.33,balanced:0.33",
    )
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="easy",
        help="Difficulty for custom --families entries.",
    )
    args = parser.parse_args()
    if args.num_instances < 200:
        raise ValueError("Each family must contain at least 200 instances")
    families = DEFAULT_FAMILIES
    if args.families:
        families = {}
        for item in args.families.split(";"):
            name, distribution = item.split("=", 1)
            families[name.strip()] = {
                "distribution": distribution.strip(),
                "difficulty": args.difficulty,
            }

    generator = Path(__file__).with_name("generate_synthetic_dag.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator": str(generator),
        "num_instances": args.num_instances,
        "seed": args.seed,
        "structure_distribution": args.structure_distribution,
        "families": {},
    }
    for offset, (name, spec) in enumerate(families.items()):
        distribution = spec["distribution"]
        difficulty = spec["difficulty"]
        output = args.output_dir / f"{name}.json"
        command = [
            sys.executable,
            str(generator),
            "--num-instances",
            str(args.num_instances),
            "--seed",
            str(args.seed + offset),
            "--output",
            str(output),
            "--budget-distribution",
            distribution,
            "--difficulty",
            difficulty,
            "--structure-distribution",
            args.structure_distribution,
            "--family-name",
            name,
        ]
        subprocess.run(command, check=True)
        manifest["families"][name] = {
            "path": str(output),
            "budget_distribution": distribution,
            "difficulty": difficulty,
            "seed": args.seed + offset,
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(families)} threshold families and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
