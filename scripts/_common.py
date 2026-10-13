"""
Common utilities for all figure scripts.
Reads deduplicated Table 2 and provides numpy arrays ready to plot.
"""

import re
from pathlib import Path
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Standardize plot style across all figures
rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
})

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


def normalize_name(raw):
    """Canonicalize NGC 4258, NGC4258, NGC04258 -> NGC4258"""
    if not raw or not raw.strip():
        return None
    s = raw.strip().upper()
    m = re.match(r"^([A-Z]+)\s*(.*)$", s)
    if not m:
        return s
    prefix, rest = m.group(1), m.group(2).strip()
    if prefix in ("NGC", "IC", "UGC", "PGC"):
        m2 = re.match(r"^0*(\d+)([A-Z]?)$", rest)
        if m2:
            return f"{prefix}{int(m2.group(1))}{m2.group(2)}"
    if prefix == "ESO":
        m2 = re.match(r"^0*(\d+)-?G?0*(\d+)$", rest)
        if m2:
            return f"ESO{int(m2.group(1))}-{int(m2.group(2))}"
    return f"{prefix}{rest}"


def parse_num(s):
    s = s.strip()
    if not s or all(c == '-' for c in s):
        return None
    try:
        return float(s)
    except:
        return None


def load_table2_dedup():
    """
    Load deduplicated Table 2.
    Returns a list of dict rows (one per unique galaxy).
    Byte offsets per CDS ReadMe of VII/296.
    """
    path = DATA / "table2_dedup.dat"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) < 100:
                continue
            name = line[0:26].strip()
            norm = normalize_name(name)
            if norm is None:
                continue
            rows.append({
                "name":   name,
                "norm":   norm,
                "BMAG":   parse_num(line[61:67]),
                "TType":  parse_num(line[76:80]),
                "Vrot":   parse_num(line[87:92]),
                "Sigmac": parse_num(line[100:105]),
                "z":      parse_num(line[114:124]),
                "Dist":   parse_num(line[125:132]),
                "A1":     parse_num(line[134:139]),
                "A2":     parse_num(line[142:148]),
            })
    return rows


def load_table1_mstar():
    """
    Load Table 1 (which has stellar masses).
    logMstar is at bytes 155-159 per CDS ReadMe (0-indexed: 154-159).
    """
    path = DATA / "table1.dat"
    t1 = {}
    with open(path) as f:
        for line in f:
            if len(line) < 100:
                continue
            # extract name (first field, bytes 1-11)
            n14 = line[0:14]
            idx = n14.find("  ")
            raw = n14[:idx].strip() if idx > 0 else n14.strip()
            key = normalize_name(raw)
            if key is None:
                continue
            t1[key] = {
                "raw":      raw,
                "TType":    parse_num(line[51:55]),
                "logMstar": parse_num(line[154:159]),
            }
    return t1


def load_tng50():
    """Load TNG50 catalog from TNGGalaxyLab (CSV)."""
    import pandas as pd
    df = pd.read_csv(DATA / "catalog_TNG50.csv")
    df["logMstar"] = np.log10(df["M_star_Msun"])
    df["A1"] = df["A1_bootstrap_mean"]
    df["A2"] = df["A2_bootstrap_mean"]
    return df


# Precomputed arrays for convenience
def arrays(exclude_extreme=True):
    """
    Return a dict of clean numpy arrays extracted from deduplicated Table 2.
    If exclude_extreme=True, drops entries with A1>1 or A2>1.
    """
    rows = load_table2_dedup()
    out = {"rows": rows}

    def get1(key):
        vals = np.array([r[key] for r in rows if r[key] is not None])
        if exclude_extreme and key in ("A1", "A2"):
            vals = vals[vals <= 1.0]
        return vals

    def getpair(k1, k2):
        pairs = [(r[k1], r[k2]) for r in rows if r[k1] is not None and r[k2] is not None]
        arr = np.array(pairs)
        if arr.size == 0:
            return np.array([]), np.array([])
        if exclude_extreme:
            if k1 in ("A1", "A2"):
                arr = arr[arr[:, 0] <= 1.0]
            if k2 in ("A1", "A2"):
                arr = arr[arr[:, 1] <= 1.0]
        return arr[:, 0], arr[:, 1]

    out["A1"] = get1("A1")
    out["A2"] = get1("A2")
    # Joint
    out["A1_j"], out["A2_j"] = getpair("A1", "A2")
    # Physical
    out["A1_T"], out["TType_1"] = getpair("A1", "TType")
    out["A2_T"], out["TType_2"] = getpair("A2", "TType")
    out["A1_B"], out["BMAG"]    = getpair("A1", "BMAG")
    out["A1_V"], out["Vrot"]    = getpair("A1", "Vrot")
    out["A1_s"], out["Sigmac"]  = getpair("A1", "Sigmac")
    out["A1_z"], out["zred"]    = getpair("A1", "z")
    return out


if __name__ == "__main__":
    arr = arrays()
    print(f"Unique galaxies (dedup):   {len(arr['rows'])}")
    print(f"With A1 (clean):           {len(arr['A1'])}")
    print(f"With A2 (clean):           {len(arr['A2'])}")
    print(f"Joint A1,A2:               {len(arr['A1_j'])}")
    print(f"A1 + T-type:               {len(arr['A1_T'])}")
    print(f"A1 + BMAG:                 {len(arr['A1_B'])}")
    print(f"A1 + Vrot:                 {len(arr['A1_V'])}")
    print(f"A1 + Sigmac:               {len(arr['A1_s'])}")
    print(f"A1 + z:                    {len(arr['A1_z'])}")
