# -*- coding: utf-8 -*-
"""
TES BUG 2 & BUG 5 - konsistensi bundle model yang diserahkan File 05 -> File 06.

Bug 2: File 05 melaporkan "model terbaik = X" tetapi menyimpan Random Forest
       secara hardcoded (`joblib.dump({"model": rf, ...})`), sehingga SHAP/LIME
       di File 06 menganalisis model yang salah.
Bug 5: File 06 mengkalibrasi LIME dengan X_test (40 baris) padahal seharusnya
       data training. Perbaikannya butuh X_train ikut tersimpan di bundle.

Cara pakai:
    python tests/test_bundle.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "files")

NAMA_KELAS = {
    "Random Forest": "RandomForestClassifier",
    "XGBoost": "XGBClassifier",
    "Logistic Regression": "LogisticRegression",
}


def main():
    bundle = joblib.load(os.path.join(DATA, "classifier_for_shap.joblib"))
    cmp_df = pd.read_csv(os.path.join(DATA, "model_comparison.csv"))

    gagal = []
    print("=" * 74)
    print("TES BUG 2 & 5 - KONSISTENSI BUNDLE MODEL")
    print("=" * 74)

    # ---------------------------------------------------------------- Bug 2
    juara = cmp_df.loc[cmp_df["macro_f1"].idxmax(), "model"]
    print(f"\nModel juara menurut model_comparison.csv : {juara}")

    if "model_name" not in bundle:
        gagal.append("bundle tidak mencatat 'model_name' -> tidak bisa diaudit")
        tercatat = "(tidak ada)"
    else:
        tercatat = bundle["model_name"]
    print(f"Model yang dicatat di bundle             : {tercatat}")

    kelas_asli = bundle["model"].__class__.__name__
    print(f"Kelas objek model yang benar-benar disimpan: {kelas_asli}")

    if tercatat == juara:
        print("OK   Bug 2a: nama model di bundle == juara perbandingan")
    else:
        gagal.append(f"nama model di bundle ('{tercatat}') != juara ('{juara}')")

    diharapkan = NAMA_KELAS.get(juara)
    if diharapkan is None:
        print(f"WARN nama model '{juara}' tidak dikenal, pemeriksaan kelas dilewati")
    elif kelas_asli == diharapkan:
        print(f"OK   Bug 2b: objek tersimpan memang {diharapkan}, bukan model lain")
    else:
        gagal.append(
            f"objek tersimpan {kelas_asli}, padahal juara '{juara}' seharusnya {diharapkan} "
            "-> ini bug hardcoded 'rf' yang lama"
        )

    # ---------------------------------------------------------------- Bug 5
    print()
    for kunci in ("X_train", "y_train", "X_test", "y_test"):
        if kunci in bundle:
            print(f"OK   bundle memuat {kunci:8s} ({len(bundle[kunci])} baris)")
        else:
            gagal.append(f"bundle tidak memuat '{kunci}' -> LIME tidak bisa pakai data training")

    if "X_train" in bundle and "X_test" in bundle:
        n_tr, n_te = len(bundle["X_train"]), len(bundle["X_test"])
        if n_tr > n_te:
            print(f"OK   Bug 5: X_train ({n_tr}) lebih besar dari X_test ({n_te}) -- "
                  "LIME punya data kalibrasi yang benar")
        else:
            gagal.append(f"X_train ({n_tr}) tidak lebih besar dari X_test ({n_te}) -- mencurigakan")

        # train dan test tidak boleh tumpang tindih (bukti split benar-benar terpisah)
        idx_tr = set(np.asarray(bundle["X_train"].index))
        idx_te = set(np.asarray(bundle["X_test"].index))
        tumpang = idx_tr & idx_te
        if not tumpang:
            print(f"OK   train/test tidak tumpang tindih (0 baris beririsan)")
        else:
            gagal.append(f"{len(tumpang)} baris muncul di train DAN test -> split bocor")

    # ---------------------------------------------------------------- imputasi
    print()
    if "X_train" in bundle:
        n_nan_tr = int(np.asarray(pd.isna(bundle["X_train"])).sum())
        n_nan_te = int(np.asarray(pd.isna(bundle["X_test"])).sum())
        if n_nan_tr == 0 and n_nan_te == 0:
            print("OK   Bug 4: X_train & X_test sudah terimputasi penuh (0 NaN), "
                  "imputer di-fit pada training")
        else:
            gagal.append(f"masih ada NaN (train={n_nan_tr}, test={n_nan_te})")

    # -------------------------------------------------- label himpunan fitur
    fset = bundle.get("feature_set", "")
    print(f"\nLabel himpunan fitur di bundle: {fset!r}")
    if "BUKAN prediksi" in fset or "konsistensi" in fset.lower():
        print("OK   Bug 1: bundle menandai eksplisit bahwa set ini BUKAN prediksi")
    else:
        gagal.append("bundle tidak menandai bahwa SET B bukan himpunan prediksi")

    # kolom IF_* yang bocor tidak boleh ada lagi di fitur model
    bocor = [c for c in bundle["X_columns"]
             if c in ("IF_VfV_FCNC", "IF_FCNC_ID", "IF_VfV_ID")]
    if not bocor:
        print("OK   Bug 1: 3 fitur IF_* yang merekonstruksi skor sudah dibuang dari fitur model")
    else:
        gagal.append(f"fitur bocor masih terpakai: {bocor}")

    print("\n" + "=" * 74)
    if gagal:
        print("TES GAGAL:")
        for g in gagal:
            print("  -", g)
        return 1
    print("SEMUA TES LOLOS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
