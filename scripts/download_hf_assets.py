"""Download Hugging Face assets on a machine with internet access."""

from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    """Download model assets for synchronization to JH.

    Args:
        None.

    Returns:
        None.
    """
    parser = ArgumentParser()
    parser.add_argument("--repo-id", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument(
        "--output-dir",
        default="data/assets/hf_models/ast-finetuned-audioset-10-10-0.4593",
    )
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo_id, local_dir=destination, local_dir_use_symlinks=False)
    print(f"downloaded {args.repo_id} to {destination}")


if __name__ == "__main__":
    main()
