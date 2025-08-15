
import numpy as np
import pandas as pd

# Kyte-Doolittle hydrophobicity scale
KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
}

# Charged residues
CHARGE = {"D": -1, "E": -1, "K": +1, "R": +1, "H": +0.1}  # H partially positive around physiological pH

# Side-chain polarity (very coarse: polar vs nonpolar vs aromatic)
POLAR = set(list("STNQYCHKRDE"))
AROM = set(list("FWYH"))

# Approximate side-chain volumes (A^3) (small set; relative only)
VOL = {
    "G": 60, "A": 88, "S": 89, "C": 108, "P": 112, "T": 117, "D": 111, "N": 114,
    "E": 138, "Q": 143, "H": 153, "M": 162, "I": 166, "L": 166, "K": 168,
    "R": 173, "F": 189, "Y": 193, "W": 227, "V": 140
}

# Grantham distances (subset + symmetric completion)
_GRANTHAM = {
"A":{"A":0,"R":112,"N":111,"D":126,"C":195,"Q":91,"E":107,"G":60,"H":86,"I":94,"L":96,"K":106,"M":84,"F":113,"P":27,"S":99,"T":58,"W":148,"Y":112,"V":64},
"R":{"A":112,"R":0,"N":86,"D":96,"C":180,"Q":43,"E":54,"G":125,"H":29,"I":97,"L":102,"K":26,"M":91,"F":97,"P":103,"S":110,"T":71,"W":101,"Y":77,"V":96},
"N":{"A":111,"R":86,"N":0,"D":23,"C":139,"Q":46,"E":42,"G":80,"H":68,"I":149,"L":153,"K":94,"M":142,"F":158,"P":91,"S":46,"T":65,"W":174,"Y":143,"V":133},
"D":{"A":126,"R":96,"N":23,"D":0,"C":154,"Q":61,"E":45,"G":94,"H":81,"I":168,"L":172,"K":101,"M":160,"F":177,"P":108,"S":65,"T":85,"W":181,"Y":160,"V":152},
"C":{"A":195,"R":180,"N":139,"D":154,"C":0,"Q":154,"E":170,"G":159,"H":174,"I":198,"L":198,"K":202,"M":196,"F":205,"P":169,"S":112,"T":149,"W":215,"Y":194,"V":192},
"Q":{"A":91,"R":43,"N":46,"D":61,"C":154,"Q":0,"E":29,"G":87,"H":24,"I":109,"L":113,"K":53,"M":101,"F":116,"P":76,"S":68,"T":42,"W":130,"Y":99,"V":96},
"E":{"A":107,"R":54,"N":42,"D":45,"C":170,"Q":29,"E":0,"G":98,"H":40,"I":134,"L":138,"K":56,"M":126,"F":140,"P":93,"S":80,"T":65,"W":152,"Y":122,"V":121},
"G":{"A":60,"R":125,"N":80,"D":94,"C":159,"Q":87,"E":98,"G":0,"H":98,"I":135,"L":138,"K":127,"M":127,"F":153,"P":42,"S":56,"T":59,"W":184,"Y":147,"V":109},
"H":{"A":86,"R":29,"N":68,"D":81,"C":174,"Q":24,"E":40,"G":98,"H":0,"I":94,"L":99,"K":32,"M":87,"F":100,"P":77,"S":89,"T":47,"W":115,"Y":83,"V":84},
"I":{"A":94,"R":97,"N":149,"D":168,"C":198,"Q":109,"E":134,"G":135,"H":94,"I":0,"L":5,"K":102,"M":10,"F":21,"P":95,"S":89,"T":89,"W":61,"Y":33,"V":29},
"L":{"A":96,"R":102,"N":153,"D":172,"C":198,"Q":113,"E":138,"G":138,"H":99,"I":5,"L":0,"K":107,"M":15,"F":22,"P":98,"S":92,"T":92,"W":61,"Y":36,"V":32},
"K":{"A":106,"R":26,"N":94,"D":101,"C":202,"Q":53,"E":56,"G":127,"H":32,"I":102,"L":107,"K":0,"M":95,"F":102,"P":103,"S":121,"T":78,"W":110,"Y":85,"V":97},
"M":{"A":84,"R":91,"N":142,"D":160,"C":196,"Q":101,"E":126,"G":127,"H":87,"I":10,"L":15,"K":95,"M":0,"F":28,"P":87,"S":74,"T":81,"W":67,"Y":36,"V":21},
"F":{"A":113,"R":97,"N":158,"D":177,"C":205,"Q":116,"E":140,"G":153,"H":100,"I":21,"L":22,"K":102,"M":28,"F":0,"P":114,"S":155,"T":103,"W":40,"Y":22,"V":50},
"P":{"A":27,"R":103,"N":91,"D":108,"C":169,"Q":76,"E":93,"G":42,"H":77,"I":95,"L":98,"K":103,"M":87,"F":114,"P":0,"S":74,"T":38,"W":147,"Y":110,"V":68},
"S":{"A":99,"R":110,"N":46,"D":65,"C":112,"Q":68,"E":80,"G":56,"H":89,"I":89,"L":92,"K":121,"M":74,"F":155,"P":74,"S":0,"T":58,"W":177,"Y":144,"V":124},
"T":{"A":58,"R":71,"N":65,"D":85,"C":149,"Q":42,"E":65,"G":59,"H":47,"I":89,"L":92,"K":78,"M":81,"F":103,"P":38,"S":58,"T":0,"W":128,"Y":92,"V":69},
"W":{"A":148,"R":101,"N":174,"D":181,"C":215,"Q":130,"E":152,"G":184,"H":115,"I":61,"L":61,"K":110,"M":67,"F":40,"P":147,"S":177,"T":128,"W":0,"Y":37,"V":88},
"Y":{"A":112,"R":77,"N":143,"D":160,"C":194,"Q":99,"E":122,"G":147,"H":83,"I":33,"L":36,"K":85,"M":36,"F":22,"P":110,"S":144,"T":92,"W":37,"Y":0,"V":55},
"V":{"A":64,"R":96,"N":133,"D":152,"C":192,"Q":96,"E":121,"G":109,"H":84,"I":29,"L":32,"K":97,"M":21,"F":50,"P":68,"S":124,"T":69,"W":88,"Y":55,"V":0}
}

# BLOSUM62 matrix (log-odds scores)
_BLOSUM62 = {
"A": {"A":4,"R":-1,"N":-2,"D":-2,"C":0,"Q":-1,"E":-1,"G":0,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S":1,"T":0,"W":-3,"Y":-2,"V":0},
"R": {"A":-1,"R":5,"N":0,"D":-2,"C":-3,"Q":1,"E":0,"G":-2,"H":0,"I":-3,"L":-2,"K":2,"M":-1,"F":-3,"P":-2,"S":-1,"T":-1,"W":-3,"Y":-2,"V":-3},
"N": {"A":-2,"R":0,"N":6,"D":1,"C":-3,"Q":0,"E":0,"G":0,"H":1,"I":-3,"L":-3,"K":0,"M":-2,"F":-3,"P":-2,"S":1,"T":0,"W":-4,"Y":-2,"V":-3},
"D": {"A":-2,"R":-2,"N":1,"D":6,"C":-3,"Q":0,"E":2,"G":-1,"H":-1,"I":-3,"L":-4,"K":-1,"M":-3,"F":-3,"P":-1,"S":0,"T":-1,"W":-4,"Y":-3,"V":-3},
"C": {"A":0,"R":-3,"N":-3,"D":-3,"C":9,"Q":-3,"E":-4,"G":-3,"H":-3,"I":-1,"L":-1,"K":-3,"M":-1,"F":-2,"P":-3,"S":-1,"T":-1,"W":-2,"Y":-2,"V":-1},
"Q": {"A":-1,"R":1,"N":0,"D":0,"C":-3,"Q":5,"E":2,"G":-2,"H":0,"I":-3,"L":-2,"K":1,"M":0,"F":-3,"P":-1,"S":0,"T":-1,"W":-2,"Y":-1,"V":-2},
"E": {"A":-1,"R":0,"N":0,"D":2,"C":-4,"Q":2,"E":5,"G":-2,"H":0,"I":-3,"L":-3,"K":1,"M":-2,"F":-3,"P":-1,"S":0,"T":-1,"W":-3,"Y":-2,"V":-2},
"G": {"A":0,"R":-2,"N":0,"D":-1,"C":-3,"Q":-2,"E":-2,"G":6,"H":-2,"I":-4,"L":-4,"K":-2,"M":-3,"F":-3,"P":-2,"S":0,"T":-2,"W":-2,"Y":-3,"V":-3},
"H": {"A":-2,"R":0,"N":1,"D":-1,"C":-3,"Q":0,"E":0,"G":-2,"H":8,"I":-3,"L":-3,"K":-1,"M":-2,"F":-1,"P":-2,"S":-1,"T":-2,"W":-2,"Y":2,"V":-3},
"I": {"A":-1,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-3,"E":-3,"G":-4,"H":-3,"I":4,"L":2,"K":-3,"M":1,"F":0,"P":-3,"S":-2,"T":-1,"W":-3,"Y":-1,"V":3},
"L": {"A":-1,"R":-2,"N":-3,"D":-4,"C":-1,"Q":-2,"E":-3,"G":-4,"H":-3,"I":2,"L":4,"K":-2,"M":2,"F":0,"P":-3,"S":-2,"T":-1,"W":-2,"Y":-1,"V":1},
"K": {"A":-1,"R":2,"N":0,"D":-1,"C":-3,"Q":1,"E":1,"G":-2,"H":-1,"I":-3,"L":-2,"K":5,"M":-1,"F":-3,"P":-1,"S":0,"T":-1,"W":-3,"Y":-2,"V":-2},
"M": {"A":-1,"R":-1,"N":-2,"D":-3,"C":-1,"Q":0,"E":-2,"G":-3,"H":-2,"I":1,"L":2,"K":-1,"M":5,"F":0,"P":-2,"S":-1,"T":-1,"W":-1,"Y":-1,"V":1},
"F": {"A":-2,"R":-3,"N":-3,"D":-3,"C":-2,"Q":-3,"E":-3,"G":-3,"H":-1,"I":0,"L":0,"K":-3,"M":0,"F":6,"P":-4,"S":-2,"T":-2,"W":1,"Y":3,"V":-1},
"P": {"A":-1,"R":-2,"N":-2,"D":-1,"C":-3,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-3,"L":-3,"K":-1,"M":-2,"F":-4,"P":7,"S":-1,"T":-1,"W":-4,"Y":-3,"V":-2},
"S": {"A":1,"R":-1,"N":1,"D":0,"C":-1,"Q":0,"E":0,"G":0,"H":-1,"I":-2,"L":-2,"K":0,"M":-1,"F":-2,"P":-1,"S":4,"T":1,"W":-3,"Y":-2,"V":-2},
"T": {"A":0,"R":-1,"N":0,"D":-1,"C":-1,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S":1,"T":5,"W":-2,"Y":-2,"V":0},
"W": {"A":-3,"R":-3,"N":-4,"D":-4,"C":-2,"Q":-2,"E":-3,"G":-2,"H":-2,"I":-3,"L":-2,"K":-3,"M":-1,"F":1,"P":-4,"S":-3,"T":-2,"W":11,"Y":2,"V":-3},
"Y": {"A":-2,"R":-2,"N":-2,"D":-3,"C":-2,"Q":-1,"E":-2,"G":-3,"H":2,"I":-1,"L":-1,"K":-2,"M":-1,"F":3,"P":-3,"S":-2,"T":-2,"W":2,"Y":7,"V":-1},
"V": {"A":0,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-2,"E":-2,"G":-3,"H":-3,"I":3,"L":1,"K":-2,"M":1,"F":-1,"P":-2,"S":-2,"T":0,"W":-3,"Y":-1,"V":4}
}

def blosum62(a, b):
    if a not in _BLOSUM62 or b not in _BLOSUM62[a]:
        return 0
    return _BLOSUM62[a][b]

def grantham(a, b):
    if a not in _GRANTHAM or b not in _GRANTHAM[a]:
        return 0
    return _GRANTHAM[a][b]

def aa_valid(a):
    return isinstance(a, str) and len(a)==1 and a.isalpha() and a.upper() in KD

def build_features(df: pd.DataFrame):
    """
    Expect columns: ref_aa, alt_aa, pos (optional), disease (optional), conservation (optional), gene (optional)
    Returns numeric feature matrix and the enriched df.
    """
    df = df.copy()
    for col in ["ref_aa","alt_aa"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    # Basic clean
    df["ref_aa"] = df["ref_aa"].str.upper()
    df["alt_aa"] = df["alt_aa"].str.upper()

    def h(a): return KD.get(a, np.nan)
    def charge(a): return CHARGE.get(a, 0.0)
    def polar(a): return 1.0 if a in POLAR else 0.0
    def arom(a): return 1.0 if a in AROM else 0.0
    def vol(a): return VOL.get(a, np.nan)

    # Compute pairwise descriptors
    df["kd_ref"] = df["ref_aa"].map(h)
    df["kd_alt"] = df["alt_aa"].map(h)
    df["kd_diff"] = df["kd_alt"] - df["kd_ref"]

    df["charge_ref"] = df["ref_aa"].map(charge)
    df["charge_alt"] = df["alt_aa"].map(charge)
    df["charge_diff"] = df["charge_alt"] - df["charge_ref"]

    df["polar_ref"] = df["ref_aa"].map(polar)
    df["polar_alt"] = df["alt_aa"].map(polar)
    df["polar_change"] = df["polar_alt"] - df["polar_ref"]

    df["arom_ref"] = df["ref_aa"].map(arom)
    df["arom_alt"] = df["alt_aa"].map(arom)
    df["arom_change"] = df["arom_alt"] - df["arom_ref"]

    df["vol_ref"] = df["ref_aa"].map(vol)
    df["vol_alt"] = df["alt_aa"].map(vol)
    df["vol_diff"] = df["vol_alt"] - df["vol_ref"]

    df["grantham"] = [grantham(a,b) if aa_valid(a) and aa_valid(b) else np.nan
                      for a,b in zip(df["ref_aa"], df["alt_aa"])]
    df["blosum62"] = [blosum62(a,b) if aa_valid(a) and aa_valid(b) else np.nan
                      for a,b in zip(df["ref_aa"], df["alt_aa"])]

    # Conservation if provided (higher = more conserved)
    if "conservation" in df.columns:
        df["cons"] = pd.to_numeric(df["conservation"], errors="coerce")
    else:
        df["cons"] = np.nan

    # Position normalized if sequence length provided
    if "pos" in df.columns and "seq_len" in df.columns:
        df["pos_norm"] = pd.to_numeric(df["pos"], errors="coerce") / pd.to_numeric(df["seq_len"], errors="coerce")
    elif "pos" in df.columns:
        df["pos_norm"] = pd.to_numeric(df["pos"], errors="coerce")
    else:
        df["pos_norm"] = np.nan

    # Feature set
    features = ["kd_diff","charge_diff","polar_change","arom_change","vol_diff","grantham","blosum62","cons","pos_norm"]
    X = df[features]
    return X, features, df
