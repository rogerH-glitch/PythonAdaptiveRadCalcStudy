from pathlib import Path
from src.cli_results import save_results
from types import SimpleNamespace
import csv

def test_csv_appends_and_has_timestamp(tmp_path):
    args = SimpleNamespace(method="adaptive", outdir=str(tmp_path))
    r1 = {"method":"adaptive","vf":0.1}
    r2 = {"method":"adaptive","vf":0.2}
    p1 = save_results(r1, args)
    p2 = save_results(r2, args)
    assert p1 == p2 and p1.exists()
    f = p1.open("r", encoding="utf-8")
    header = f.readline().strip().split(",")
    f.close()
    assert header[0] == "timestamp"
    rows = list(csv.DictReader(p1.open("r", encoding="utf-8")))
    assert len(rows) == 2 and "timestamp" in rows[0]
    assert rows[0]["vf"] == "0.1" and rows[1]["vf"] == "0.2"
