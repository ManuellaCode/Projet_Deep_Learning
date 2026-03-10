from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

DEFAULT_DATASET = "paultimothymooney/chest-xray-pneumonia"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge le dataset Chest X-Ray (Pneumonia) depuis Kaggle."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Slug Kaggle (owner/dataset).",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Dossier de destination.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Conserver l'archive zip après extraction.",
    )
    return parser.parse_args()


def validate_kaggle_credentials() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "kaggle.json introuvable. Place ton fichier API ici: "
            f"{kaggle_json}"
        )


def has_expected_structure(root: Path) -> bool:
    return all((root / split).exists() for split in ["train", "val", "test"])


def locate_dataset_root(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "chest_xray",
        output_dir / "chest_xray" / "chest_xray",
        output_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and has_expected_structure(candidate):
            return candidate
    return None


def download_and_extract(dataset_slug: str, output_dir: Path, keep_zip: bool) -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi

    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(output_dir), quiet=False)

    zip_files = sorted(output_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(
            "Aucune archive zip trouvée après téléchargement Kaggle."
        )

    zip_path = zip_files[0]
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    dataset_root = locate_dataset_root(output_dir)
    if dataset_root is None:
        raise FileNotFoundError(
            "Extraction terminée, mais structure train/val/test introuvable."
        )

    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    return dataset_root


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).resolve()

    validate_kaggle_credentials()
    dataset_root = download_and_extract(args.dataset, output_dir, args.keep_zip)

    print(f"Dataset prêt: {dataset_root}")
    print("Conseil: garde ce dossier hors versioning Git (déjà géré par .gitignore).")


if __name__ == "__main__":
    main()
