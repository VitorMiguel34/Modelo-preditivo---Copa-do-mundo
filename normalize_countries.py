#!/usr/bin/env python3
"""
Script para normalizar nomes de países em múltiplos ficheiros CSV sob a diretoria `dados/`.

Funcionalidades:
- Percorre recursivamente `./dados` e encontra todos os `.csv`.
- Detecta colunas que provavelmente têm nomes de países por heurísticas sobre o nome da coluna.
- Normaliza os nomes usando `pycountry` e um dicionário de mapeamento para variações comuns.
- Guarda os CSVs processados em `./dados_processados` preservando a subpasta relativa.

Uso:
    python normalize_countries.py --input ./dados --output ./dados_processados

"""
from pathlib import Path
import os
import argparse
import logging
from typing import List, Optional

import pandas as pd
try:
    import pycountry
except Exception:
    pycountry = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


COUNTRY_COLUMN_KEYWORDS = [
    "country",
    "pais",
    "nation",
    "team",
    "country_code",
    "countrycode",
    "country name",
    "country_name",
    "countryname",
    "teamname",
    "home_team",
    "away_team",
]

# Small mapping for very common variants and abbreviations
MANUAL_MAP = {
    "u.s.": "United States",
    "u.s.a.": "United States",
    "usa": "United States",
    "us": "United States",
    "u.k.": "United Kingdom",
    "uk": "United Kingdom",
    "england": "England",
    "scotland": "Scotland",
    "wales": "Wales",
    "northern ireland": "Northern Ireland",
    "korea, south": "South Korea",
    "korea, north": "North Korea",
    "iran, islamic republic of": "Iran",
    "czech republic": "Czechia",
}


def find_csv_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*.csv"):
        if p.is_file():
            files.append(p)
    return files


def is_country_column_name(col: str) -> bool:
    key = col.strip().lower()
    # check exact keywords or containment
    for kw in COUNTRY_COLUMN_KEYWORDS:
        if kw in key:
            return True
    return False


def normalize_country_name(value: Optional[str]) -> Optional[str]:
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s == "":
        return s
    key = s.lower()
    # manual mapping first
    if key in MANUAL_MAP:
        return MANUAL_MAP[key]

    # remove punctuation often used in abbreviations
    simple = key.replace(".", "").replace("/", " ").strip()

    # try alpha-2, alpha-3
    if pycountry:
        try:
            c = pycountry.countries.get(alpha_2=s.upper())
            if c:
                return c.name
        except Exception:
            pass
        try:
            c = pycountry.countries.get(alpha_3=s.upper())
            if c:
                return c.name
        except Exception:
            pass

        # fuzzy search if available
        try:
            results = pycountry.countries.search_fuzzy(s)
            if results:
                return results[0].name
        except Exception:
            pass

    # fallback: title-case the cleaned value
    return simple.title()


def process_file(path: Path, out_root: Path, input_root: Path) -> None:
    logging.info(f"Processing: {path}")
    # try reading with utf-8, fallback to latin-1
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="latin-1")

    cols_to_normalize = [c for c in df.columns if is_country_column_name(c)]

    if not cols_to_normalize:
        logging.info(f"  No obvious country columns found in {path.name}; skipping normalization.")
    else:
        logging.info(f"  Columns to normalize: {cols_to_normalize}")
        for col in cols_to_normalize:
            df[col] = df[col].apply(normalize_country_name)

    # build output path preserving relative structure
    rel = path.relative_to(input_root)
    out_path = out_root.joinpath(rel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info(f"  Saved processed file to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Normalize country names in CSV files under a directory")
    parser.add_argument("--input", "-i", default="./dados", help="Input root directory (default: ./dados)")
    parser.add_argument("--output", "-o", default="./dados_processados", help="Output root directory")
    parser.add_argument("--dry-run", action="store_true", help="Only list files that would be processed")
    args = parser.parse_args()

    if pycountry is None:
        logging.warning("pycountry not found. Install with `pip install pycountry` for best results.")

    input_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()

    if not input_root.exists():
        logging.error(f"Input directory not found: {input_root}")
        return

    csv_files = find_csv_files(input_root)
    logging.info(f"Found {len(csv_files)} CSV files under {input_root}")
    if args.dry_run:
        for p in csv_files:
            logging.info(f"  Would process: {p}")
        return

    for p in csv_files:
        try:
            process_file(p, out_root, input_root)
        except Exception as e:
            logging.error(f"Failed to process {p}: {e}")


if __name__ == "__main__":
    main()
