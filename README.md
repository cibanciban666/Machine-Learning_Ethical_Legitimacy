# EHSL-ML+ — Pipeline Analitik Disertasi

**Ethical-Governance Survey-driven and Human-supervised Learning Plus**
Disertasi Nanan Soekarna

Pipeline analitik 8 tahap untuk mengukur persepsi narapidana terhadap perlakuan etis aparat penegak hukum di Indonesia — dari pembersihan data, validasi instrumen, konstruksi indeks komposit (MAS/ECI/NREI), analisis sensitivitas, machine learning, sampai ekspor dashboard.

> ### ⚠ Status data
> Repo ini memuat **data dummy** hasil simulasi `np.random` (File 00). **Tidak ada responden sungguhan.** Seluruh angka bersifat ilustratif untuk mendemonstrasikan pipeline.
>
> Sebelum memasukkan data riil, **baca peringatan privasi di `.gitignore`** — riwayat git bersifat permanen dan data narapidana berpotensi teridentifikasi.

---

## Mulai cepat

```bash
cd files

for nb in 01_validity_reliability 02_index_construction 03_sensitivity_analysis \
          04_feature_engineering 05_ml_analytics 06_explainability_erm 07_dashboard_export; do
  python -m nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=1500 $nb.ipynb
done

cd .. && python tests/test_leakage.py && python tests/test_bundle.py
```

Seluruh pipeline selesai sekitar **2 menit**. Kedua tes harus berakhir dengan `SEMUA TES LOLOS`.

Panduan lengkap — prasyarat, tiga cara menjalankan, peta ketergantungan, penanganan masalah — ada di **[`CARA_MENJALANKAN.md`](CARA_MENJALANKAN.md)**.

---

## Struktur

| Tahap | Notebook | Fungsi |
|---|---|---|
| 00 | `00_data_cleaning.ipynb` | Pembersihan: reverse coding, duplikat, *missing value*, *straight-lining* |
| 01 | `01_validity_reliability.ipynb` | Cronbach's alpha, EFA (promax + parallel analysis) |
| 02 | `02_index_construction.ipynb` | Normalisasi & indeks komposit MAS / ECI / NREI |
| 03 | `03_sensitivity_analysis.ipynb` | Monte Carlo, Sobol GSA, Morris screening |
| 04 | `04_feature_engineering.ipynb` | Fitur interaksi, temporal, ECV |
| 05 | `05_ml_analytics.ipynb` | Clustering, klasifikasi, deteksi anomali |
| 06 | `06_explainability_erm.ipynb` | SHAP, LIME, Ethical Reinforcement Mechanism |
| 07 | `07_dashboard_export.ipynb` | Ekspor struktur 5-layer dashboard |

```
files/     8 notebook + CSV keluaran
tests/     2 tes regresi (kebocoran data & konsistensi model)
```

---

## Catatan penting saat mengutip hasil

Pipeline ini memakai **tiga himpunan fitur dengan label tegas**, dan hanya satu yang sah disebut "prediksi":

| Himpunan | Boleh disebut |
|---|---|
| **SET A** | demonstrasi kebocoran data — bukan hasil |
| **SET B** | pemeriksaan konsistensi internal — **bukan** prediksi |
| **SET C** | **prediksi** (satu-satunya yang sah) |

Angka yang sah dikutip ada di `files/feature_set_comparison.csv`, khususnya kolom `boleh_disebut`. Latar belakangnya dijelaskan di [`LAPORAN_PERBAIKAN.md`](LAPORAN_PERBAIKAN.md) bagian Bug 1.

Notebook juga **memeriksa dirinya sendiri**: audit kebocoran berjalan otomatis tiap kali dijalankan, dan `assert` akan menggagalkan notebook bila himpunan fitur "bersih" ternyata tercemar.

---

## Batas etis

Seluruh keluaran pipeline ini adalah **alat bantu interpretasi berbasis persepsi dengan supervisi manusia** (*Human-in-the-Loop*).

**Bukan** bukti hukum, **bukan** alat menentukan kesalahan individu, dan **bukan** alat memeringkat atau menghukum aktor maupun institusi. Skor rendah pada aktor tertentu menandakan *area yang perlu direview*, bukan vonis.

---

## Dokumentasi

| Berkas | Isi |
|---|---|
| [`CARA_MENJALANKAN.md`](CARA_MENJALANKAN.md) | Panduan operasional: prasyarat, cara jalan, urutan, troubleshooting |
| [`LAPORAN_PERBAIKAN.md`](LAPORAN_PERBAIKAN.md) | 17 perbaikan bug: masalah, solusi, justifikasi, metrik sebelum/sesudah |
| `Review_Metodologi_Notebook_EHSL-ML+.docx` | Review metodologi awal (teknis) |
| `Penjelasan_Sederhana_Masalah_dan_Perbaikan.docx` | Penjelasan masalah dalam bahasa non-teknis |

---

## Yang masih menunggu di luar coding

1. **Data riil** — ganti File 00 dengan pemuatan data kuesioner (lihat `CARA_MENJALANKAN.md` bagian 11)
2. **Expert panel** untuk menetapkan bobot ERM final; saat ini keempat skema dinyatakan setara
3. **Keputusan item reduction** berdasarkan `files/item_reduction_report.csv` — tidak boleh murni statistik
4. **Visualisasi dashboard** dari `files/dashboard_data.json`
