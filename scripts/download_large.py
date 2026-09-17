"""Descarga masiva xeno-canto de Columba livia (API v3)."""
import json, os, sys, time
from pathlib import Path
import urllib.request

KEY = os.environ.get("XENO_KEY") or "PONER_KEY"
BASE = "https://xeno-canto.org/api/3/recordings"

def fetch(q, page=1):
    url = f"{BASE}?query={q}&page={page}&key={KEY}"
    req = urllib.request.Request(url, headers={"User-Agent": "paloma-pi-v2/0.1"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def main():
    outdir = Path("data/audio_xc_large")
    outdir.mkdir(parents=True, exist_ok=True)

    all_recs = []
    for page in range(1, 200):
        try:
            d = fetch("sp:%22Columba%20livia%22", page=page)
        except Exception as e:
            print(f"page {page}: {e}")
            break
        recs = d.get("recordings", [])
        if not recs:
            break
        all_recs.extend(recs)
        print(f"page {page}: {len(recs)} recs (total {len(all_recs)})")
        if len(recs) < 500:
            break
        time.sleep(1)

    print("metadata total:", len(all_recs))
    Path("data/xeno_canto_metadata_large.json").write_text(json.dumps(all_recs, indent=2))

    # Calidad A/B y diversidad de recordistas
    pool = [r for r in all_recs if r.get("q") in ("A", "B")]
    print("calidad A/B:", len(pool))

    import random
    random.seed(42)
    random.shuffle(pool)
    chosen = {}
    for r in pool:
        rec = r.get("rec", "?")
        chosen.setdefault(rec, []).append(r)

    chosen_list = []
    for rec, recs in chosen.items():
        for r in recs[:5]:
            if len(chosen_list) >= 100:
                break
            chosen_list.append(r)
        if len(chosen_list) >= 100:
            break
    print("elegidas:", len(chosen_list))

    ok = 0
    for i, r in enumerate(chosen_list):
        fname = outdir / f"{r['id']}.mp3"
        if fname.exists() and fname.stat().st_size > 5000:
            ok += 1
            continue
        try:
            urllib.request.urlretrieve(r["file"], str(fname))
            ok += 1
            if i % 10 == 0:
                print(f"  {i+1}/{len(chosen_list)} descargadas")
            time.sleep(0.5)
        except Exception as e:
            print("  falló", r["id"], e)
    print("descargadas:", ok)
    Path("data/chosen_100.json").write_text(json.dumps(chosen_list, indent=2))

if __name__ == "__main__":
    main()
