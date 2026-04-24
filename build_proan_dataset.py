#!/usr/bin/env python3
from pathlib import Path
import csv
import re

INPUT_FASTA = Path("human_proteins.fasta")
OUTPUT_CSV = Path("human_proteins_clean.csv")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def parse_fasta(path):
    header = None
    seq_parts = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

    if header is not None:
        yield header, "".join(seq_parts)

def extract_id(header):
    m = re.match(r'^(?:sp|tr)\|([^|]+)\|', header)
    if m:
        return m.group(1)
    return header.split()[0].split("|")[0]

def clean_sequence(seq):
    seq = seq.upper().replace(" ", "").replace("\t", "")
    seq = re.sub(r"[^A-Z]", "", seq)
    seq = "".join(aa for aa in seq if aa in VALID_AA)
    return seq

def main():
    if not INPUT_FASTA.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_FASTA}")

    seen_ids = set()
    rows = []

    for header, seq in parse_fasta(INPUT_FASTA):
        protein_id = extract_id(header)
        if protein_id in seen_ids:
            continue
        seen_ids.add(protein_id)

        seq = clean_sequence(seq)
        if len(seq) < 20:
            continue

        rows.append([protein_id, seq, len(seq)])

    rows.sort(key=lambda x: x[2], reverse=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Sequence", "Length"])
        writer.writerows(rows)

    print(f"Готово: {OUTPUT_CSV}")
    print(f"Записей: {len(rows)}")
    if rows:
        lengths = [r[2] for r in rows]
        print(f"Мин длина: {min(lengths)}")
        print(f"Макс длина: {max(lengths)}")

if __name__ == "__main__":
    main()
