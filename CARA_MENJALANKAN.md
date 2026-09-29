# Cara Menjalankan Pipeline EHSL-ML+

**Disertasi Nanan Soekarna**
Panduan operasional: prasyarat, cara menjalankan, urutan, dan penanganan masalah.

Dokumen ini adalah panduan **praktis**. Untuk penjelasan *apa* yang diperbaiki dan *kenapa*, lihat `LAPORAN_PERBAIKAN.md`.

---

## Daftar Isi

- [Ringkasan cepat](#ringkasan-cepat)
- [1. Prasyarat](#1-prasyarat)
- [2. Struktur folder](#2-struktur-folder)
- [3. Aturan wajib sebelum menjalankan](#3-aturan-wajib-sebelum-menjalankan)
- [4. Cara A — VS Code](#4-cara-a--vs-code-paling-nyaman)
- [5. Cara B — Terminal](#5-cara-b--terminal-paling-cepat)
- [6. Cara C — Google Colab](#6-cara-c--google-colab)
- [7. Urutan & peta ketergantungan](#7-urutan--peta-ketergantungan)
- [8. Menjalankan tes](#8-menjalankan-tes)
- [9. Daftar keluaran](#9-daftar-keluaran)
- [10. Penanganan masalah](#10-penanganan-masalah)
- [11. Saat data riil sudah terkumpul](#11-saat-data-riil-sudah-terkumpul)
- [12. Memulai dari nol di komputer lain](#12-memulai-dari-nol-di-komputer-lain)

---

## Ringkasan cepat

Kalau Anda hanya ingin menjalankan semuanya sekarang dan tidak mau membaca panjang-panjang:

```bash
cd "C:/Disertasi baru Pak Nanan/Coding/files"

for nb in 01_validity_reliability 02_index_construction 03_sensitivity_analysis \
          04_feature_engineering 05_ml_analytics 06_explainability_erm 07_dashboard_export; do
  echo "### $nb"
  python -m nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=1500 $nb.ipynb
done
```

Lalu periksa hasilnya:

```bash
cd "C:/Disertasi baru Pak Nanan/Coding"
python tests/test_leakage.py
python tests/test_bundle.py
```

Kedua tes harus berakhir dengan **`SEMUA TES LOLOS`**.

Tiga hal yang paling sering menyebabkan gagal:

1. **Folder kerja salah** — harus di dalam `files/`, bukan di folder `Coding`
2. **Urutan dibolak-balik** — 04 wajib sebelum 05, 05 sebelum 06, 06 sebelum 07
3. **Lupa `--ExecutePreprocessor.timeout`** — batas bawaan `nbconvert` hanya 30 detik per sel, dan sel terberat File 05 melewatinya

---

## 1. Prasyarat

### Yang sudah terpasang di komputer ini

Seluruh kebutuhan **sudah dipasang** saat proses perbaikan. Tidak ada yang perlu di-*install* lagi.

```
Python 3.14.2
Lokasi: C:\Users\cici\AppData\Local\Programs\Python\Python314\python.exe
```

| Pustaka | Versi | Dipakai di |
|---|---|---|
| `numpy` | 2.4.6 | semua notebook |
| `pandas` | 3.0.3 | semua notebook |
| `scipy` | 1.18.1 | 00, 05 |
| `scikit-learn` | 1.9.1 | 05 |
| `xgboost` | 3.4.1 | 05 |
| `shap` | 0.52.0 | 06 |
| `lime` | 0.2.0.1 | 06 |
| `SALib` | 1.6.0 | 03 |
| `factor-analyzer` | 0.5.1 | 01 |
| `joblib` | 1.6.0 | 05, 06 |
| `nbconvert` | 7.17.1 | untuk menjalankan dari terminal |
| `ipykernel` | 7.3.0 | untuk menjalankan di VS Code |
| `matplotlib` | 3.11.2 | (tersedia, belum dipakai) |

### Yang TIDAK terpasang

`jupyterlab` dan `notebook` **tidak** dipasang, jadi perintah `jupyter lab` atau `jupyter notebook` **tidak akan jalan**.

Ini bukan masalah — ada dua jalan lain yang sudah siap (VS Code dan terminal). Tapi kalau Anda memang ingin antarmuka Jupyter klasik di browser:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

### VS Code

Sudah terpasang beserta semua ekstensi yang dibutuhkan:

```
ms-python.python              ms-toolsai.jupyter
ms-python.vscode-pylance      ms-toolsai.jupyter-renderers
ms-python.debugpy             ms-toolsai.jupyter-keymap
```

Kernel `python3` juga sudah terdaftar dan menunjuk ke Python 3.14 yang berisi seluruh pustaka di atas. Jadi notebook bisa langsung dijalankan di VS Code tanpa persiapan apa pun.

### Spesifikasi komputer

Tidak ada kebutuhan khusus. Datanya kecil (199 responden, 896 baris aktor), semua berjalan di CPU biasa, tidak perlu GPU. Kebutuhan RAM di bawah 1 GB.

---

## 2. Struktur folder

```
C:\Disertasi baru Pak Nanan\Coding\
│
├── files/                    ← KODE AKTIF. Semua pekerjaan di sini.
│   ├── 00_data_cleaning.ipynb
│   ├── 01_validity_reliability.ipynb
│   ├── ...
│   ├── 07_dashboard_export.ipynb
│   └── (23 berkas CSV / JSON / model hasil keluaran)
│
├── tests/                    ← 2 skrip pemeriksa bug
│   ├── test_leakage.py
│   └── test_bundle.py
│
├── files_backup_pre_fix/     ← arsip kondisi SEBELUM diperbaiki. JANGAN dijalankan.
├── _before_runs/             ← bukti angka "sebelum". Tidak dipakai pipeline.
│
├── CARA_MENJALANKAN.md       ← dokumen ini
├── LAPORAN_PERBAIKAN.md      ← apa yang diperbaiki & kenapa
├── Review_Metodologi_Notebook_EHSL-ML+.docx
├── Penjelasan_Sederhana_Masalah_dan_Perbaikan.docx
└── files.zip                 ← arsip lama (isi versi bermasalah, boleh diabaikan)
```

**Yang penting diingat:** seluruh notebook ada di `files/`, dan hanya di situ. Folder `Coding` sendiri tidak lagi memuat notebook.

---

## 3. Aturan wajib sebelum menjalankan

### ✅ Aturan 1 — Folder kerja harus `files/`

Notebook membaca dan menulis CSV memakai **nama berkas polos** tanpa path, contohnya:

```python
df_actor = pd.read_csv("data_actor_clean.csv")
```

Artinya berkas itu dicari di **folder tempat notebook dijalankan**. Kalau folder kerjanya salah, hasilnya:

```
FileNotFoundError: [Errno 2] No such file or directory: 'data_actor_clean.csv'
```

- **Di terminal:** selalu `cd` ke `files/` lebih dulu
- **Di VS Code:** buka folder `files` sebagai *workspace* (File → Open Folder → pilih `files`), bukan folder `Coding`

### ✅ Aturan 2 — Urutan tidak boleh dibolak-balik

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

Yang paling kritis: **04 sebelum 05**, **05 sebelum 06**, **06 sebelum 07**. Rinciannya di [bagian 7](#7-urutan--peta-ketergantungan).

### ✅ Aturan 3 — File 00 tidak perlu dijalankan

File 00 **membangkitkan ulang data dummy dari nol**. Berkas hasilnya (`data_actor_clean.csv` dan `data_respondent_clean.csv`) sudah tersedia di `files/`.

Kalau Anda tetap menjalankannya, **seluruh notebook sesudahnya wajib dijalankan ulang** karena datanya berubah. Notebook ini memakai *seed* tetap sehingga hasilnya seharusnya sama persis — tapi tidak ada gunanya mengambil risiko tanpa alasan.

File 00 baru relevan nanti, saat isinya diganti dengan data kuesioner riil (lihat [bagian 11](#11-saat-data-riil-sudah-terkumpul)).

### ⛔ Aturan 4 — Jangan menjalankan apa pun di `files_backup_pre_fix/`

Notebook menulis CSV ke folder tempat ia berada. Menjalankan notebook di folder arsip akan **menimpa berkas aslinya**, dan folder itu berhenti menjadi "kondisi sebelum" yang murni.

Kalau Anda perlu menjalankan versi lama untuk membandingkan, **salin dulu** ke folder baru:

```bash
cp -r files_backup_pre_fix files_uji_lama
cd files_uji_lama
# jalankan di sini, arsip aslinya tetap aman
```

---

## 4. Cara A — VS Code (paling nyaman)

Paling cocok kalau Anda ingin **melihat hasil sel demi sel**, mengubah kode, atau memeriksa tabel keluaran.

### Langkah

1. Buka **VS Code**
2. **File → Open Folder** → pilih `C:\Disertasi baru Pak Nanan\Coding\files`
3. Klik notebook di panel kiri, misalnya `05_ml_analytics.ipynb`
4. Di kanan atas klik **Select Kernel**
   - Pilih **Python Environments** → **Python 3.14.2** (yang di `...\Programs\Python\Python314`)
   - Kalau muncul tawaran *"Install ipykernel"*, abaikan — sudah terpasang
5. Jalankan:
   - **Run All** (tombol ▶▶ di atas) untuk seluruh notebook
   - atau `Shift+Enter` untuk satu sel, lalu lanjut ke sel berikutnya

### Kelebihan

- Hasil tiap sel langsung terlihat, tabel bisa di-*scroll*
- Mudah mengubah kode lalu menjalankan ulang hanya sel yang terdampak
- Error muncul lengkap dengan lokasinya

### Yang perlu diperhatikan

- **Jangan lupa menyimpan** (`Ctrl+S`) setelah menjalankan, kalau ingin hasilnya tersimpan di notebook
- Kalau notebook dijalankan sebagian-sebagian, variabel dari sel sebelumnya harus sudah dijalankan. Kalau ragu, pakai **Restart** lalu **Run All**

---

## 5. Cara B — Terminal (paling cepat)

Paling cocok kalau Anda ingin **menjalankan semuanya sekaligus** tanpa klik-klik.

### Semua notebook sekaligus

```bash
cd "C:/Disertasi baru Pak Nanan/Coding/files"

for nb in 01_validity_reliability 02_index_construction 03_sensitivity_analysis \
          04_feature_engineering 05_ml_analytics 06_explainability_erm 07_dashboard_export; do
  echo "### menjalankan $nb"
  python -m nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=1500 $nb.ipynb
done
```

### Satu notebook saja

```bash
cd "C:/Disertasi baru Pak Nanan/Coding/files"
python -m nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=1500 05_ml_analytics.ipynb
```

### Arti tiap opsi

| Opsi | Arti |
|---|---|
| `--to notebook` | keluarannya tetap berupa notebook (bukan HTML/PDF) |
| `--execute` | jalankan selnya, bukan hanya mengonversi |
| `--inplace` | tulis hasilnya ke notebook yang sama — **tanpa ini** akan dibuat berkas baru |
| `--ExecutePreprocessor.timeout=1500` | batas waktu per sel 1500 detik. **Wajib** — bawaannya hanya 30 detik dan File 05 pasti melewatinya |

Menjalankan ketujuh notebook memakan waktu sekitar **2 menit** seluruhnya (rincian di [bagian 7](#7-urutan--peta-ketergantungan)).

### Kalau ingin hasilnya tidak menimpa notebook asli

Hilangkan `--inplace` dan beri nama keluaran:

```bash
python -m nbconvert --to notebook --execute --output "hasil_05.ipynb" \
       --ExecutePreprocessor.timeout=1500 05_ml_analytics.ipynb
```

---

## 6. Cara C — Google Colab

Bisa, dan inilah cara yang Anda pakai sebelumnya. Perlu dua langkah tambahan.

### Langkah

1. Buka [colab.research.google.com](https://colab.research.google.com), unggah notebook yang ingin dijalankan

2. **Unggah juga berkas CSV-nya** — ini yang sering terlewat. Notebook mencari CSV di folder yang sama, jadi berkas masukan yang dibutuhkan (lihat [bagian 7](#7-urutan--peta-ketergantungan)) harus ada di sesi Colab. Cara paling praktis: unggah seluruh isi `files/` lewat panel *Files* di kiri.

3. Tambahkan sel paling atas untuk memasang pustaka yang tidak ada di Colab:

```python
!pip install -q factor_analyzer SALib lime shap xgboost
```

4. Jalankan seperti biasa

### Catatan khusus Colab

- **Shim di File 01 otomatis tidak aktif** di Colab, karena versi `scikit-learn`-nya masih lama sehingga argumen `force_all_finite` masih dikenali. Tidak perlu diapa-apakan — notebook dirancang jalan di kedua lingkungan.
- **Sesi Colab hilang saat idle.** Berkas yang dihasilkan akan lenyap kalau tidak diunduh. Setelah selesai, unduh CSV keluarannya lewat panel *Files*.
- **Angka bisa berbeda tipis** dari hasil di komputer ini, karena versi pustakanya berbeda. Yang berbeda hanya angka di desimal belakang, bukan kesimpulannya.

---

## 7. Urutan & peta ketergantungan

Tabel ini menjawab pertanyaan: *"kalau saya mengubah notebook X, mana saja yang harus dijalankan ulang?"*

| Notebook | Membaca | Menulis |
|---|---|---|
| **00** Data Cleaning | *(tidak ada — membangkitkan data)* | `data_actor_clean.csv`, `data_respondent_clean.csv` |
| **01** Validitas & Reliabilitas | `data_actor_clean.csv`, `data_respondent_clean.csv` | `reliability_summary.csv`, `item_reduction_report.csv` |
| **02** Index Construction | `data_actor_clean.csv`, `data_respondent_clean.csv` | `data_index_actor.csv`, `data_index_respondent.csv` |
| **03** Sensitivity Analysis | `data_index_respondent.csv` | `sensitivity_summary.csv`, `sensitivity_sobol.csv`, `sensitivity_sobol_bounds_lama.csv`, `sensitivity_morris.csv` |
| **04** Feature Engineering | `data_index_actor.csv`, `data_index_respondent.csv`, `data_respondent_clean.csv` | `feature_matrix.csv` |
| **05** ML Analytics | `feature_matrix.csv`, `data_index_respondent.csv`, `data_index_actor.csv` | `ml_results.csv`, `model_comparison.csv`, `feature_set_comparison.csv`, `objective_report.csv`, `classifier_for_shap.joblib` |
| **06** Explainability + ERM | `classifier_for_shap.joblib`, `data_index_respondent.csv`, `data_index_actor.csv` | `shap_global_importance.csv`, `erm_scores.csv`, `erm_by_actor.csv`, `erm_by_stage.csv`, `explainability_erm_summary.csv` |
| **07** Dashboard Export | hampir semua CSV di atas | `dashboard_data.json`, `dashboard_respondent_level.csv` |

### Aturan praktis "jalankan ulang dari mana"

| Kalau Anda mengubah... | Jalankan ulang |
|---|---|
| **File 00** (atau ganti data riil) | **semuanya**: 01 → 07 |
| File 01 | hanya 01 *(tidak ada yang bergantung padanya)* |
| File 02 | 02 → 03 → 04 → 05 → 06 → 07 |
| File 03 | 03, lalu 07 |
| File 04 | 04 → 05 → 06 → 07 |
| File 05 | 05 → 06 → 07 |
| File 06 | 06, lalu 07 |
| File 07 | hanya 07 |

Perhatikan **File 01 adalah cabang mati** — ia hanya menghasilkan laporan reliabilitas dan tidak dipakai notebook lain. Jadi mengubah File 01 tidak memaksa apa pun dijalankan ulang.

### Perkiraan waktu jalan

Diukur di komputer ini (Python 3.14, CPU biasa):

| Notebook | Waktu terukur | Kenapa |
|---|---|---|
| 01 | 22 detik | parallel analysis 500 iterasi + 2 kali EFA |
| 02 | 6 detik | hanya aritmetika |
| 03 | 9 detik | Monte Carlo 10.000 iterasi + Sobol 2×5.120 evaluasi + Morris |
| 04 | 8 detik | pivot & agregasi |
| 05 | **47 detik** | 7 × repeated CV (50 evaluasi masing-masing) = ~350 pelatihan model |
| 06 | 14 detik | SHAP + LIME |
| 07 | 6 detik | hanya merakit CSV |
| **Total** | **112 detik (~2 menit)** | |

Angka di atas **hasil pengukuran sungguhan** di komputer ini, bukan perkiraan. Tiap angka sudah termasuk ~3–5 detik untuk menyalakan kernel Python.

File 05 yang paling lama (47 detik), dan itu **disengaja**: metrik headline-nya kini dari *repeated cross-validation* (5 lipatan × 10 ulangan) agar angkanya tepercaya, bukan dari satu kali pembagian data yang hasilnya bisa berubah karena kebetulan.

---

## 8. Menjalankan tes

Terpisah dari notebook, dan cepat (beberapa detik).

```bash
cd "C:/Disertasi baru Pak Nanan/Coding"
python tests/test_leakage.py
python tests/test_bundle.py
```

### Apa yang diperiksa

**`test_leakage.py`** — memeriksa kebocoran data. Skrip ini sengaja *mencoba menyontek*: ia berusaha menghitung balik skor VfV/FCNC/ID dari fitur yang diberikan ke model. Kalau berhasil, berarti kunci jawaban masih bocor.

Ia memastikan dua hal:
- himpunan fitur lama **memang** bocor → membuktikan bug-nya nyata
- **SET C bersih** → membuktikan perbaikannya bekerja

**`test_bundle.py`** — memeriksa serah-terima model dari File 05 ke File 06:
- model yang disimpan sama dengan juara yang dilaporkan?
- `X_train` tersedia untuk kalibrasi LIME?
- data training & test benar-benar terpisah (tidak beririsan)?
- masih ada data kosong (`NaN`) tersisa?
- fitur `IF_*` yang bocor sudah keluar dari daftar?

### Kapan harus dijalankan

- **Setelah** menjalankan File 04 dan 05 — karena `test_bundle.py` memeriksa berkas yang dihasilkan File 05
- **Setiap kali** Anda mengubah daftar fitur di File 05
- **Wajib** saat data riil pertama kali masuk

### Keluaran yang benar

Keduanya harus berakhir dengan:

```
SEMUA TES LOLOS
```

Kalau ada yang gagal, keluarannya akan menyebutkan persis apa yang salah, contohnya:

```
TES GAGAL:
  - SET C masih bocor, korelasi 0.997
```

Itu tanda ada fitur baru yang diam-diam membawa komponen target. **Jangan diabaikan** — justru itulah gunanya tes ini.

---

## 9. Daftar keluaran

Setelah pipeline selesai, `files/` akan memuat berkas berikut. Ini yang dipakai menulis Bab IV.

### Data antara

| Berkas | Isi |
|---|---|
| `data_actor_clean.csv` | data bersih level responden × aktor (896 baris) |
| `data_respondent_clean.csv` | data bersih level responden (199 baris) |
| `data_index_actor.csv` | + skor MAS/ECI/NREI per aktor |
| `data_index_respondent.csv` | + skor keseluruhan per responden |
| `feature_matrix.csv` | matriks fitur untuk ML (199 baris × 71 kolom) |

### Hasil analisis

| Berkas | Isi | Catatan penting |
|---|---|---|
| `reliability_summary.csv` | Cronbach's alpha tiap konstruk | |
| `item_reduction_report.csv` | item yang perlu ditinjau | keputusan akhir tetap di tangan Anda + expert panel |
| `sensitivity_summary.csv` | flip-rate, Spearman, rasio ST | |
| `sensitivity_sobol.csv` | Sobol dengan anggaran **adil** | **ini yang dipakai** |
| `sensitivity_sobol_bounds_lama.csv` | Sobol versi lama (timpang) | hanya pembanding, jangan dikutip |
| `sensitivity_morris.csv` | Morris screening | |
| **`feature_set_comparison.csv`** | akurasi per himpunan fitur + kolom `boleh_disebut` | **inilah angka ML yang sah dikutip** |
| `model_comparison.csv` | perbandingan 3 model | dari satu kali split, **jangan** dijadikan angka utama |
| `objective_report.csv` | komponen `L_total`, ARI cluster, status kestabilan | |
| `ml_results.csv` | cluster & skor anomali per responden | |
| `classifier_for_shap.joblib` | model terlatih + data train/test | masukan File 06 |
| `shap_global_importance.csv` | peringkat pengaruh fitur | |
| `erm_scores.csv` | skor ERM per responden | |
| `erm_by_actor.csv` | ERM per kategori aktor | |
| `erm_by_stage.csv` | ERM per tahap KUHAP | |
| `explainability_erm_summary.csv` | ringkasan | |
| `dashboard_data.json` | struktur 5-layer untuk visualisasi | |
| `dashboard_respondent_level.csv` | data per responden untuk dashboard | |

> ⚠ **Kolom `boleh_disebut` di `feature_set_comparison.csv` penting.** Ia menandai himpunan mana yang sah disebut "prediksi" (hanya SET C) dan mana yang **tidak** (SET A dan SET B). Baca kolom itu sebelum mengutip angka akurasi ke Bab IV. Penjelasan lengkapnya ada di `LAPORAN_PERBAIKAN.md` bagian Bug 1.

---

## 10. Penanganan masalah

### `FileNotFoundError: 'data_actor_clean.csv'`

**Penyebab:** folder kerja salah — notebook dijalankan dari luar `files/`.

**Solusi:**
- Terminal: `cd "C:/Disertasi baru Pak Nanan/Coding/files"` lebih dulu
- VS Code: tutup *workspace*, lalu **File → Open Folder** → pilih folder `files`

---

### `FileNotFoundError: 'feature_matrix.csv'` (atau `classifier_for_shap.joblib`)

**Penyebab:** notebook dijalankan tidak berurutan. Berkas ini dihasilkan notebook sebelumnya.

**Solusi:** jalankan notebook pendahulunya lebih dulu. Lihat tabel di [bagian 7](#7-urutan--peta-ketergantungan). Contoh: `feature_matrix.csv` dihasilkan File 04, jadi jalankan 04 sebelum 05.

---

### `TimeoutError` atau sel berhenti setelah 30 detik

**Penyebab:** lupa memberi `--ExecutePreprocessor.timeout`. Batas bawaan `nbconvert` hanya 30 detik **per sel**, sementara sel terberat File 05 (repeated CV) melewatinya.

Nilai 1500 detik yang dipakai di panduan ini jauh lebih besar daripada kebutuhan sebenarnya (File 05 seluruhnya hanya ~47 detik) — sengaja diberi kelonggaran besar supaya tetap aman di komputer yang lebih lambat atau saat data riil yang lebih besar masuk.

**Solusi:** tambahkan opsinya:

```bash
python -m nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=1500 05_ml_analytics.ipynb
```

---

### `KeyError: 'C:\\...\\joblib_memmapping_folder_...'` saat File 05

**Ini BUKAN kegagalan.** Abaikan saja.

Pesan ini muncul dari pembersihan berkas sementara Windows setelah pemrosesan paralel (`n_jobs=-1`) selesai. Bisa muncul beberapa kali berurutan dan tampak menakutkan karena berbentuk *traceback*.

**Cara memastikan runnya sukses:** cari baris terakhir ini —

```
[NbConvertApp] Writing 54649 bytes to 05_ml_analytics.ipynb
```

Kalau baris itu ada, notebooknya berhasil ditulis dan hasilnya sah.

---

### `TypeError: check_array() got an unexpected keyword argument 'force_all_finite'`

**Penyebab:** pustaka `factor_analyzer` 0.5.1 (rilis terakhir yang ada) masih memakai argumen yang sudah dihapus di `scikit-learn` ≥ 1.8.

**Sudah ditangani** — File 01 memuat *shim* kompatibilitas di sel pertama yang menerjemahkan argumen lama ke yang baru. Kalau error ini tetap muncul, berarti sel pertama belum dijalankan.

**Solusi:** jalankan File 01 dari sel paling atas (**Restart** lalu **Run All**), jangan mulai dari tengah.

---

### `UserWarning: X does not have valid feature names` saat File 06

**Bisa diabaikan.** Muncul karena LIME mengirim data dalam bentuk array tanpa nama kolom. Tidak memengaruhi hasil.

---

### `TqdmWarning: IProgress not found`

**Bisa diabaikan.** Hanya soal tampilan bilah kemajuan di terminal. Tidak memengaruhi hasil.

---

### `NameError: name 'Xtr' is not defined` (atau variabel lain)

**Penyebab:** notebook dijalankan sebagian-sebagian, dan sel yang mendefinisikan variabel itu belum dijalankan di sesi ini.

**Solusi:** **Restart** kernel lalu **Run All**. Notebook dirancang untuk dijalankan dari atas ke bawah secara berurutan.

---

### `AssertionError: SET C seharusnya bersih dari komponen target!`

**Ini alarm yang bekerja sebagaimana mestinya** — bukan kerusakan.

Artinya ada fitur yang masuk ke SET C padahal membawa komponen skor NREI, sehingga hasil "prediksi" jadi tidak sah.

**Solusi:** periksa daftar `set_c_cols` di File 05 sel 7. Isinya harus **hanya** variabel demografi/legal: `age`, `gender_encoded`, `education_encoded`, `case_type_encoded`, `sentence_length_months`, `time_served_months`, `institution_encoded`, `region_encoded`. Kalau ada nama lain yang mengandung `VfV`, `FCNC`, `ID`, `ECV`, `IF_`, `T_`, `MAS`, `ECI`, `NREI`, atau `GR` — keluarkan.

---

### Hasil angkanya beda dari yang ada di laporan

**Kemungkinan normal** kalau:
- dijalankan di Colab atau komputer lain (versi pustaka berbeda)
- File 00 dijalankan ulang (data dummy dibangkitkan lagi)

Yang berbeda biasanya hanya desimal belakang, bukan kesimpulannya. Yang **harus** tetap sama:
- SET C tidak mengalahkan baseline
- kedua tes lolos
- jumlah faktor EFA = 3

Kalau salah satu dari tiga hal itu berubah, ada yang perlu diperiksa.

---

## 11. Saat data riil sudah terkumpul

Inilah tujuan akhirnya. Yang perlu diganti **hanya File 00**; struktur enam notebook sesudahnya tidak berubah.

### Langkah

**1. Ganti isi File 00.** Hapus bagian yang membangkitkan data dummy (Bagian 1 dan 2), ganti dengan memuat data riil:

```python
df_respondent = pd.read_csv("kuesioner_responden.csv")
df_actor      = pd.read_csv("kuesioner_aktor.csv")
```

Bagian pembersihan (reverse coding, duplikat, nilai di luar skala, *missing value*, *straight-lining*) **jangan diubah** — justru bagian itulah yang baru benar-benar berguna pada data riil.

**2. Pastikan kolomnya sama persis.** Data riil harus memuat kolom dengan nama identik:

- `df_respondent`: `respondent_id`, `age`, `gender`, `education`, `case_type`, `sentence_length_months`, `time_served_months`, `institution`, `region`, `GR1`–`GR5`
- `df_actor`: `respondent_id`, `actor_category`, `VfV1`–`VfV5`, `FCNC1`–`FCNC5`, `ID1`–`ID5`

`actor_category` harus berisi salah satu dari: `Investigator`, `Prosecutor`, `Judge`, `Advocate`, `CorrectionalOfficer`.

**3. Jalankan seluruh pipeline** dari 00 sampai 07.

**4. Jalankan kedua tes.** Ini **wajib**, bukan opsional — fitur baru dari data riil bisa membawa kebocoran yang tidak ada di data dummy.

### Yang perlu diperiksa ulang pada data riil

| Hal | Yang diharapkan | Kalau tidak sesuai |
|---|---|---|
| Cronbach's alpha | 0.70–0.85 | alpha > 0.90 wajar dicurigai item redundan |
| Jumlah faktor EFA | 3 | kalau bukan 3, struktur konstruk perlu ditinjau ulang |
| Korelasi antar-faktor | dilaporkan apa adanya | |
| SET C vs baseline | **bisa jadi menang!** | kalau menang, itu **temuan sungguhan** — demografi memang memprediksi persepsi governance |
| Kestabilan cluster (ARI) | ≥ 0.50 agar layak dilaporkan | di bawah itu, jangan sebut tipologi |
| Flip-rate sensitivity | < 5% | di atas itu, laporkan sebagai keterbatasan |
| Skema bobot ERM | mungkin tidak lagi seri | kalau ada yang menonjol, tetap tunggu expert panel |

Perhatikan baris **SET C**: pada data dummy ia kalah dari baseline, dan itu benar karena datanya acak. Pada data riil, kalau SET C mengalahkan baseline, itu **hasil positif yang sah dan layak dilaporkan** — bukti bahwa profil demografis/legal betul-betul berkaitan dengan persepsi keadilan.

### Yang masih menunggu di luar coding

1. **Expert panel** untuk menetapkan bobot ERM final (Fase 2 proposal). Sampai itu ada, File 06 menyatakan keempat skema setara dan memakai `equal` sebagai default netral.
2. **Keputusan item reduction** berdasarkan `item_reduction_report.csv` — proposal menegaskan ini tidak boleh murni statistik, harus disertai pertimbangan teoretis dan penilaian ahli.
3. **Membangun visualisasi** dashboard dari `dashboard_data.json` (Streamlit / Dash / Power BI).

---

## 12. Memulai dari nol di komputer lain

Kalau pipeline ini perlu dijalankan di komputer lain (misalnya laptop pembimbing atau komputer kampus):

### 1. Pasang Python

Versi 3.10 atau lebih baru, dari [python.org](https://python.org). Saat memasang, **centang "Add Python to PATH"**.

### 2. Pasang pustaka

```bash
python -m pip install numpy pandas scipy scikit-learn xgboost shap lime \
                      SALib factor_analyzer joblib nbconvert nbformat ipykernel
```

Kalau ingin antarmuka Jupyter di browser, tambahkan `jupyterlab`.

### 3. Salin folder

Salin folder `files/` dan `tests/` apa adanya. Folder `files_backup_pre_fix/` dan `_before_runs/` tidak dibutuhkan untuk menjalankan — itu hanya arsip.

### 4. Jalankan

Ikuti [bagian 5](#5-cara-b--terminal-paling-cepat).

### Catatan kompatibilitas

- **`scikit-learn` ≥ 1.8:** *shim* di File 01 akan aktif otomatis dan menangani ketidakcocokan `factor_analyzer`. Tidak perlu tindakan apa pun.
- **`scikit-learn` < 1.8:** shim otomatis tidak aktif. Juga tidak perlu tindakan apa pun.
- **`pandas` 3.x:** sudah diuji, berjalan normal.
- **Windows:** pesan `KeyError joblib_memmapping` mungkin muncul di File 05. Abaikan (lihat [bagian 10](#10-penanganan-masalah)).

---

## Lampiran — Perintah yang paling sering dipakai

```bash
# Pindah ke folder kerja (SELALU lakukan ini lebih dulu)
cd "C:/Disertasi baru Pak Nanan/Coding/files"

# Jalankan satu notebook
python -m nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=1500 05_ml_analytics.ipynb

# Jalankan rantai setelah mengubah File 04
for nb in 04_feature_engineering 05_ml_analytics 06_explainability_erm 07_dashboard_export; do
  python -m nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=1500 $nb.ipynb
done

# Periksa hasilnya
cd "C:/Disertasi baru Pak Nanan/Coding"
python tests/test_leakage.py && python tests/test_bundle.py

# Lihat angka ML yang sah dikutip
python -c "import pandas as pd; print(pd.read_csv('files/feature_set_comparison.csv').to_string(index=False))"

# Kembalikan satu notebook ke kondisi sebelum diperbaiki
cp "files_backup_pre_fix/05_ml_analytics.ipynb" "files/05_ml_analytics.ipynb"
```

---

**Dokumen terkait:**

| Berkas | Isi |
|---|---|
| `LAPORAN_PERBAIKAN.md` | apa yang diperbaiki, solusinya, justifikasinya, angka sebelum/sesudah |
| `Review_Metodologi_Notebook_EHSL-ML+.docx` | review metodologi awal (teknis) |
| `Penjelasan_Sederhana_Masalah_dan_Perbaikan.docx` | penjelasan masalah dalam bahasa awam |
