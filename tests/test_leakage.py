# -*- coding: utf-8 -*-
"""
TES BUG 1 - Kebocoran data (data leakage) pada himpunan fitur klasifikasi.

Target klasifikasi adalah kategori NREI, yang dihitung dari:
    NREI_overall = 0.40*MAS + 0.40*ECI + 0.20*GR
                 = 0.24*VfV + 0.20*FCNC + 0.36*ID + 0.20*GR   (semua skor _overall)

Tes ini mencoba MEREKONSTRUKSI VfV/FCNC/ID_overall dari fitur-fitur yang
diberikan ke model. Kalau rekonstruksi berhasil (korelasi ~1.0), berarti
kunci jawaban ikut bocor ke dalam fitur -> akurasi model tidak bermakna.

Cara pakai:
    python tests/test_leakage.py

Lolos  = jalur rekonstruksi yang sudah ditutup memang gagal.
Gagal  = masih ada jalur bocor yang terbuka.
"""
import sys
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "files")

ACTORS = ["Investigator", "Prosecutor", "Judge", "Advocate", "CorrectionalOfficer"]
CONSTRUCTS = ["VfV", "FCNC", "ID"]

# Ambang: korelasi di atas ini dianggap "bocor" (kunci jawaban bisa dihitung balik)
AMBANG_BOCOR = 0.90


def _corr(rec, true):
    """Korelasi + max abs error antara skor rekonstruksi dan skor asli."""
    ok = rec.notna() & true.notna()
    if ok.sum() < 10:
        return np.nan, np.nan
    return rec[ok].corr(true[ok]), (rec[ok] - true[ok]).abs().max()


# ---------------------------------------------------------------- jalur serangan
def serang_lewat_if(fm):
    """Jalur 1: fitur interaksi IF_* adalah perkalian skor overall.

        IF_VfV_FCNC = (VfV/100)*(FCNC/100)
        IF_FCNC_ID  = (FCNC/100)*(ID/100)
        IF_VfV_ID   = (VfV/100)*(ID/100)

    Maka  VfV = 100*sqrt(IF_VfV_FCNC * IF_VfV_ID / IF_FCNC_ID),  dst.
    """
    butuh = ["IF_VfV_FCNC", "IF_FCNC_ID", "IF_VfV_ID"]
    if not all(k in fm.columns for k in butuh):
        return None
    a, b, c = fm["IF_VfV_FCNC"], fm["IF_FCNC_ID"], fm["IF_VfV_ID"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "VfV": np.sqrt(a * c / b) * 100,
            "FCNC": np.sqrt(a * b / c) * 100,
            "ID": np.sqrt(b * c / a) * 100,
        }


def serang_lewat_ecv(fm):
    """Jalur 2: ECV_{aktor}_{konstruk} adalah skor per-aktor mentah.
    Skor overall = rata-rata skor per-aktor -> cukup dirata-ratakan."""
    out = {}
    for con in CONSTRUCTS:
        cols = [f"ECV_{a}_{con}" for a in ACTORS if f"ECV_{a}_{con}" in fm.columns]
        if not cols:
            return None
        out[con] = fm[cols].mean(axis=1, skipna=True)
    return out


def serang_lewat_pivot(fm):
    """Jalur 3: kolom pivot VfV_{aktor} juga skor per-aktor (duplikat ECV)."""
    cols = [f"VfV_{a}" for a in ACTORS if f"VfV_{a}" in fm.columns]
    if not cols:
        return None
    return {"VfV": fm[cols].mean(axis=1, skipna=True)}


def serang_lewat_temporal(fm):
    """Jalur 4: T_{konstruk}_t1..t4 adalah rata-rata skor per tahap.
    Rekonstruksi hanya hampiran (t3 menggabung 2 aktor), tapi tetap sangat erat."""
    out = {}
    for con in CONSTRUCTS:
        cols = [f"T_{con}_t{i}" for i in range(1, 5) if f"T_{con}_t{i}" in fm.columns]
        if not cols:
            return None
        out[con] = fm[cols].mean(axis=1, skipna=True)
    return out


JALUR = [
    ("IF_* (perkalian skor)", serang_lewat_if),
    ("ECV_{aktor} (rata-rata)", serang_lewat_ecv),
    ("pivot VfV_{aktor}", serang_lewat_pivot),
    ("T_*_t1..t4 (per tahap)", serang_lewat_temporal),
]


def audit(nama_set, kolom_fitur, fm, idx):
    """Coba semua jalur rekonstruksi pada satu himpunan fitur."""
    sub = fm[[c for c in kolom_fitur if c in fm.columns]].copy()
    print(f"\n  {nama_set}  ({sub.shape[1]} fitur)")
    korelasi_tertinggi = 0.0
    ada_jalur = False
    for label, fn in JALUR:
        hasil = fn(sub)
        if hasil is None:
            print(f"    {label:28s} : jalur TERTUTUP (kolom tidak tersedia)")
            continue
        ada_jalur = True
        for con, rec in hasil.items():
            r, err = _corr(rec, idx[f"{con}_overall"])
            if np.isnan(r):
                continue
            korelasi_tertinggi = max(korelasi_tertinggi, abs(r))
            tanda = "<-- BOCOR" if abs(r) >= AMBANG_BOCOR else ""
            print(f"    {label:28s} : {con:5s} korelasi={r:.6f} max_err={err:.6f} {tanda}")
    if not ada_jalur:
        print("    -> tidak ada satu pun jalur rekonstruksi yang tersedia")
    return korelasi_tertinggi


def main():
    fm = pd.read_csv(os.path.join(DATA, "feature_matrix.csv"))
    idx = pd.read_csv(os.path.join(DATA, "data_index_respondent.csv"))
    fm = fm.merge(
        idx[["respondent_id", "VfV_overall", "FCNC_overall", "ID_overall"]],
        on="respondent_id", how="left", suffixes=("", "_target"),
    )
    for con in CONSTRUCTS:
        if f"{con}_overall_target" in fm.columns:
            idx_col = fm[f"{con}_overall_target"]
        else:
            idx_col = fm[f"{con}_overall"]
        fm[f"__true_{con}"] = idx_col
    true = pd.DataFrame({f"{c}_overall": fm[f"__true_{c}"] for c in CONSTRUCTS})

    # ---- definisi himpunan fitur, mengikuti File 05 ----
    bukan_fitur = ["respondent_id", "NREI_cat", "cluster_kmeans", "cluster_hierarchical"]
    semua = [c for c in fm.columns
             if c not in bukan_fitur and not c.startswith("__true_")
             and not c.endswith("_overall_target")]

    kolom_bocor_lama = [
        "MAS_overall", "ECI_overall", "NREI_overall",
        "NREI_Investigator", "NREI_Prosecutor", "NREI_Judge",
        "NREI_Advocate", "NREI_CorrectionalOfficer",
        "VfV_overall", "FCNC_overall", "ID_overall", "GR_overall",
    ]
    set_b_lama = [c for c in semua if c not in kolom_bocor_lama]

    # SET C (perbaikan): hanya demografi/legal - nol komponen konstruk
    set_c_demografi = [
        "age", "gender_encoded", "education_encoded", "case_type_encoded",
        "sentence_length_months", "time_served_months",
        "institution_encoded", "region_encoded",
    ]

    print("=" * 78)
    print("TES BUG 1 - AUDIT KEBOCORAN HIMPUNAN FITUR")
    print("=" * 78)
    print(f"N responden: {len(fm)}   ambang bocor: korelasi >= {AMBANG_BOCOR}")

    print("\n[SEBELUM PERBAIKAN] SET B lama, yang di notebook diberi label 'leakage-safe':")
    r_lama = audit("SET B (lama)", set_b_lama, fm, true)

    print("\n[SESUDAH PERBAIKAN] SET C, fitur prediksi bersih (demografi/legal saja):")
    r_baru = audit("SET C (baru)", set_c_demografi, fm, true)

    print("\n" + "=" * 78)
    gagal = []

    # Assertion 1: SET B lama HARUS terbukti bocor (membuktikan bug itu nyata)
    if r_lama >= AMBANG_BOCOR:
        print(f"OK   Bug terbukti nyata : SET B lama bocor (korelasi tertinggi {r_lama:.6f})")
    else:
        gagal.append(f"SET B lama seharusnya terbukti bocor, tapi korelasi hanya {r_lama:.6f}")

    # Assertion 2: SET C baru TIDAK boleh bocor (membuktikan perbaikan berhasil)
    if r_baru < AMBANG_BOCOR:
        print(f"OK   Perbaikan berhasil : SET C tidak bocor (korelasi tertinggi {r_baru:.6f})")
    else:
        gagal.append(f"SET C masih bocor, korelasi {r_baru:.6f}")

    # Assertion 3: SET C tidak boleh memuat kolom turunan konstruk apa pun
    terlarang = [c for c in set_c_demografi
                 if any(k in c for k in ("VfV", "FCNC", "ID_", "ECV", "IF_", "T_",
                                         "MAS", "ECI", "NREI", "GR"))]
    if not terlarang:
        print("OK   SET C bersih       : nol kolom turunan konstruk")
    else:
        gagal.append(f"SET C memuat kolom turunan konstruk: {terlarang}")

    print("=" * 78)
    if gagal:
        print("TES GAGAL:")
        for g in gagal:
            print("  -", g)
        return 1
    print("SEMUA TES LOLOS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
