"""一次只覆盖一个变量，运行一组可复现的消融训练。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_FLAGS = {
    "dropout": "--dropout",
    "num_heads": "--num-heads",
    "num_layers": "--num-layers",
    "d_model": "--embedding-dim",
    "block_size": "--block-size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单变量消融实验。")
    parser.add_argument("--experiment", choices=EXPERIMENT_FLAGS, required=True)
    parser.add_argument(
        "--values",
        nargs="+",
        required=True,
        help="包含基线值和一个或多个变体值。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tiny_shakespeare.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "ablations",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flag = EXPERIMENT_FLAGS[args.experiment]
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else PROJECT_ROOT / args.output_root
    )

    for value in args.values:
        run_name = f"{args.experiment}-{value.replace('.', '_')}"
        checkpoint_dir = output_root / run_name
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--config",
            str(args.config.resolve()),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--skip-overfit",
            flag,
            value,
        ]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "command.txt").write_text(
                " ".join(command) + "\n",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            with (checkpoint_dir / "run.log").open(
                "w",
                encoding="utf-8",
            ) as log_file:
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)


if __name__ == "__main__":
    main()
