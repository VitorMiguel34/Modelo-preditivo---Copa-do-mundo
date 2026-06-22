#!/usr/bin/env python3
"""
Script para normalizar nomes de países em múltiplos ficheiros CSV sob a diretoria `dados/`.

Funcionalidades:
- Percorre recursivamente `./dados` e encontra todos os `.csv`.
- Detecta colunas que provavelmente têm nomes de países por heurísticas sobre o nome da coluna.
- Normaliza os nomes usando mapeamento manual (com suporte a tradução PT -> EN) e suporte opcional a pycountry.
- Guarda os CSVs processados em `./dados_processados` preservando a subpasta relativa.

Uso:
    python normalize_countries.py --input ./dados --output ./dados_processados
"""

from pathlib import Path
import os
import argparse
import logging
from typing import List, Optional
import functools
import unicodedata  

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

# Dicionário robusto unificado: Trata variações históricas, siglas E traduções do Português!
# NOTA: Todas as chaves em Português foram salvas SEM ACENTO, pois a função remove os acentos antes da busca.
MANUAL_MAP = {
    # --- TRADUÇÕES DO PORTUGUÊS PARA O INGLÊS PADRÃO ---
    "brasil": "Brazil",
    "marrocos": "Morocco",
    "marroco": "Morocco",
    "espanha": "Spain",
    "franca": "France",
    "inglaterra": "England",
    "portugal": "Portugal",
    "holanda": "Netherlands",
    "belgica": "Belgium",
    "alemanha": "Germany",
    "croacia": "Croatia",
    "colombia": "Colombia",
    "mexico": "Mexico",
    "senegal": "Senegal",
    "uruguai": "Uruguay",
    "estados unidos": "United States",
    "eua": "United States",
    "japao": "Japan",
    "suica": "Switzerland",
    "ira": "Iran",
    "turquia": "Turkey",
    "equador": "Ecuador",
    "austria": "Austria",
    "coreia do sul": "South Korea",
    "australia": "Australia",
    "argelia": "Algeria",
    "egito": "Egypt",
    "canada": "Canada",
    "noruega": "Norway",
    "costa do marfim": "Ivory Coast",
    "panama": "Panama",
    "suecia": "Sweden",
    "republica tcheca": "Czechia",
    "paraguai": "Paraguay",
    "escocia": "Scotland",
    "tunisia": "Tunisia",
    "republica democratica do congo": "DR Congo",
    "rd congo": "DR Congo",
    "uzbequistao": "Uzbekistan",
    "catar": "Qatar",
    "iraque": "Iraq",
    "africa do sul": "South Africa",
    "arabia saudita": "Saudi Arabia",
    "jordania": "Jordan",
    "bosnia e herzegovina": "Bosnia and Herzegovina",
    "cabo verde": "Cape Verde",
    "gana": "Ghana",
    "curacao": "Curacao",
    "haiti": "Haiti",
    "nova zelandia": "New Zealand",

    # --- VARIAÇÕES GERAIS E HISTÓRICAS EM INGLÊS ---
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
    
    # Coreias
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "korea, south": "South Korea",
    "korea dpr": "North Korea",
    "north korea": "North Korea",
    "korea, north": "North Korea",
    
    # Alemanha Histórica
    "west germany": "Germany",         
    "germany fr": "Germany",           
    "east germany": "East Germany",     
    "german dr": "East Germany",        
    
    # Limpezas de logs e formatos alternativos de bases internacionais
    "iran, islamic republic of": "Iran",
    "czech republic": "Czechia",
    "ir iran": "Iran",
    "côte d'ivoire": "Ivory Coast",
    "c°te d'ivoire": "Ivory Coast",  
    "cote d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "dr congo": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "yugoslavia": "Yugoslavia",
    "czechoslovakia": "Czechoslovakia",
    "republic of ireland": "Ireland",
    "united arab emirates": "United Arab Emirates",
    "soviet union": "Soviet Union",
    "ussr": "Soviet Union",
    
    # Antiga Iugoslávia / Sérvia
    "fr yugoslavia": "Yugoslavia",
    "serbia & montenegro": "Serbia and Montenegro",
    "serbia and montenegro": "Serbia and Montenegro",
    
    # China
    "china pr": "China",
    "china": "China",
}


def find_csv_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*.csv"):
        if p.is_file():
            files.append(p)
    return files


def is_country_column_name(col: str) -> bool:
    key = col.strip().lower()
    for kw in COUNTRY_COLUMN_KEYWORDS:
        if kw in key:
            return True
    return False

#FUNÇÃO GERAL DE NORMALIZAÇÃO DE NOMES DE SELEÇÕES
@functools.lru_cache(maxsize=None)
def normalize_team_name(team_name: Optional[str]) -> Optional[str]:
    """
    Função UNIFICADA de tratamento. Limpa resíduos técnicos, remove acentos,
    aplica tradução/mapeamento manual de apelidos e validação de países.
    """
    if team_name is None or pd.isna(team_name):
        return team_name
    
    team_name = str(team_name).strip()
    if not team_name:
        return team_name
    
    # 1. Forçar minúsculas e limpar sujeiras comuns de encoding html/csv
    key = team_name.lower()
    key = key.replace('"rn"">', '').replace('rn>', '').replace('"', '')
    
    # 2. REMOVER ACENTOS (Ex: "França" vira "franca", "Japão" vira "japao")
    key = "".join(
        c for c in unicodedata.normalize('NFD', key)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Casos especiais de strings parciais antes do mapeamento exato
    if "ivoire" in key or "ivory" in key:
        return "Ivory Coast"
    if "israel*" in key:
        return "Israel"
    if "bulgaria" in key:
        return "Bulgaria"
        
    # Remover prefixos técnicos comuns em datasets antigos da FIFA
    for prefix in ["ir ", "dr ", "fyr "]:
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
            
    # 3. Verificar se existe no MANUAL_MAP (que agora contém as traduções!)
    if key in MANUAL_MAP:
        return MANUAL_MAP[key]
        
    # Tratamentos adicionais de fallback
    if "republic of ireland" in key:
        return "Ireland"
    if "united arab emirates" in key:
        return "United Arab Emirates"
    if "trinidad and tobago" in key:
        return "Trinidad and Tobago"
    if "serbia and montenegro" in key:
        return "Serbia and Montenegro"
    if "yugoslavia" in key:
        return "Yugoslavia"
        
    simple_clean = key.replace(".", "").replace("/", " ").strip()
    
    # 4. Suporte ao Pycountry se instalado
    if pycountry:
        try:
            c = pycountry.countries.get(alpha_2=simple_clean.upper())
            if c: return c.name
        except Exception: pass
        try:
            c = pycountry.countries.get(alpha_3=simple_clean.upper())
            if c: return c.name
        except Exception: pass
        try:
            results = pycountry.countries.search_fuzzy(simple_clean)
            if results: return results[0].name
        except Exception: pass

    return simple_clean.title()


# Criando um alias de compatibilidade caso alguma célula antiga chame o nome anterior
normalize_country_name = normalize_team_name


def process_file(path: Path, out_root: Path, input_root: Path) -> None:
    logging.info(f"Processing: {path}")
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
            df[col] = df[col].apply(normalize_team_name)

    # Build output path preserving relative structure
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