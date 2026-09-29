# Laporan Perbaikan Bug — Pipeline EHSL-ML+

**Disertasi Nanan Soekarna**
Tanggal perbaikan: 29 September 2026
Lingkup: `files/01_validity_reliability.ipynb` s/d `files/07_dashboard_export.ipynb`

Dokumen ini adalah catatan lengkap perbaikan atas bug-bug yang ditemukan pada review sebelumnya (lihat `Review_Metodologi_Notebook_EHSL-ML+.docx` dan `Penjelasan_Sederhana_Masalah_dan_Perbaikan.docx`).

Untuk setiap bug dijelaskan lima hal, dengan pola yang sama:

1. **Apa yang salah** — kondisi kode sebelum diperbaiki
2. **Kenapa itu masalah** — dampaknya bagi keabsahan disertasi, dengan analogi sehari-hari
3. **Apa yang saya ubah** — perubahan konkret di file dan sel mana
4. **Justifikasi** — kenapa solusi ini yang dipilih, bukan yang lain
5. **Angka sebelum → sesudah** — bukti terukur bahwa bug benar-benar hilang

---

## Daftar Isi

- [Kamus istilah singkat](#kamus-istilah-singkat)
- [Ringkasan seluruh perbaikan](#ringkasan-seluruh-perbaikan)
- [Bagian I — Kebocoran data & bug logika](#bagian-i--kebocoran-data--bug-logika)
- [Bagian II — Robustness statistik](#bagian-ii--robustness-statistik)
- [Bagian III — Metodologi](#bagian-iii--metodologi)
- [Bagian IV — Temuan baru selama perbaikan](#bagian-iv--temuan-baru-selama-perbaikan)
- [Yang sengaja TIDAK diubah](#yang-sengaja-tidak-diubah)
- [Cara memverifikasi sendiri](#cara-memverifikasi-sendiri)
- [Daftar berkas](#daftar-berkas)
- [Ujung terbuka](#ujung-terbuka)

---

## Kamus istilah singkat

Beberapa istilah yang sering muncul di dokumen ini. Cukup pahami versi sederhananya.

| Istilah | Arti sederhana |
|---|---|
| **Model / machine learning** | Program yang "belajar" dari contoh data, lalu menebak jawaban untuk data baru. Seperti murid belajar dari soal latihan lalu diuji. |
| **Fitur** | Petunjuk yang diberikan ke model sebagai bahan menebak (usia, skor per aktor, jenis kasus). |
| **Target** | Jawaban yang harus ditebak model. Di sini: kategori NREI (Low/Moderate/High). |
| **Training / testing** | Data dibagi dua: sebagian untuk belajar, sebagian disembunyikan untuk ujian. |
| **Akurasi** | Persentase tebakan yang benar. 0.85 = 85% benar. |
| **Baseline** | Patokan terendah: akurasi kalau model cuma menebak kelas terbanyak terus-menerus. Model yang tidak mengalahkan baseline berarti tidak menemukan apa pun. |
| **Overfitting** | Model "menghafal" soal latihan sampai detail tak pentingnya, sehingga jeblok saat diuji soal baru. |
| **Data leakage (kebocoran)** | Kunci jawaban ikut terselip di antara petunjuk. Model terlihat pintar, padahal menyontek. |
| **Cross-validation (CV)** | Data dibagi-ulang dan diuji berkali-kali, lalu nilainya dirata-rata. Lebih tepercaya daripada satu kali ujian. |
| **SHAP** | Alat untuk mengukur fitur mana yang paling menentukan keputusan model. |
| **LIME** | Alat untuk menjelaskan kenapa satu kasus individual diputuskan begitu. |
| **EFA / factor loading** | Uji statistik apakah item kuesioner benar-benar mengukur konstruk yang dimaksud. |
| **Rotasi (varimax / promax)** | Cara "memutar sudut pandang" pada hasil EFA agar polanya lebih jelas terbaca. |

---

## Ringkasan seluruh perbaikan

**17 perbaikan** (15 dari laporan review + 2 temuan baru yang disetujui), ditambah 1 keputusan teknis yang saya ambil sendiri karena memblokir eksekusi.

| # | Bug | File | Sifat |
|---|---|---|---|
| 1 | Kebocoran struktural pada himpunan fitur | 05 | **kritis** |
| 2 | Model yang disimpan bukan model juara | 05 | bug murni |
| 3 | `L_total` mencampur dua model berbeda | 05 | bug murni |
| 4 | Imputasi dilakukan sebelum pembagian data | 05 | kebocoran |
| 5 | LIME dikalibrasi dengan data test | 06 | bug murni |
| 6 | Uji Sobol memberi "goyangan" timpang | 03 | kesimpulan salah |
| 7 | Metrik dari satu kali split (n=40) | 05 | robustness |
| 8 | XGBoost tanpa penyeimbang kelas | 05 | perbandingan timpang |
| 9 | Skor kestabilan cluster diabaikan | 05 | pelaporan |
| 10 | Jumlah anomali disebut "temuan" | 05 | pelaporan |
| 11 | Rotasi varimax + kriteria Kaiser saja | 01 | metodologi |
| 12 | Kolom fitur duplikat & rasio fitur tak sehat | 04, 05 | overfitting |
| 13 | Pemenang bobot ERM dipilih dari noise | 06 | pelaporan |
| 14 | Flag "asal centang" dibuat lalu diabaikan | 05 | tindak lanjut |
| 15 | Bobot implisit ID tidak didokumentasikan | 02 | dokumentasi |
| a | Label "turun" dicetak tanpa syarat | 05 | **temuan baru** |
| b | `IF_ID_Stage` masih mendominasi SHAP | 04, 05 | **temuan baru** |
| — | `factor_analyzer` × scikit-learn 1.9 bentrok | 01 | blocker teknis |

**Hasil akhir:** seluruh 8 notebook dijalankan ulang dari awal sampai akhir — **0 error**. Dua tes regresi baru — **keduanya lolos**.

---

## Bagian I — Kebocoran data & bug logika

Ini kelompok yang dikerjakan lebih dulu, karena bug di sini membuat **angka hasilnya salah**, bukan cuma kurang rapi.

---

### Bug 1 — Kebocoran struktural: model terlihat pintar padahal menyontek

#### Apa yang salah

Model diminta menebak kategori NREI seseorang. Kode lama sudah berusaha hati-hati: kolom `MAS_overall`, `ECI_overall`, `NREI_overall`, dan skor konstruk keseluruhan sengaja dibuang dari daftar fitur, lalu himpunan sisanya diberi nama **"SET B (safe)"**.

Ternyata label "safe" itu tidak benar. Ada **empat jalan** yang masih terbuka untuk menghitung balik kunci jawaban:

| Jalur | Cara | Hasil uji |
|---|---|---|
| Aljabar `IF_*` | `VfV = 100 × √(IF_VfV_FCNC × IF_VfV_ID / IF_FCNC_ID)` | korelasi **1.000000**, error **0.0** |
| Rata-rata `ECV_{aktor}` | skor keseluruhan = rata-rata skor per-aktor | korelasi **1.000000**, error **0.0** |
| Rata-rata pivot `VfV_{aktor}` | idem | korelasi **1.000000**, error **0.0** |
| Rata-rata `T_*_t1..t4` | rata-rata skor per tahap | korelasi **0.9928–0.9936** |

Tiga jalur pertama bukan "hampir", tapi **eksak** — error nol sampai sepuluh angka desimal.

Akar masalahnya lebih dalam dari sekadar beberapa kolom nakal: **kategori NREI itu sendiri dihitung dari jawaban kuesioner, dan hampir semua fitur juga dibuat dari jawaban kuesioner yang sama.** Kolom `ECV_{aktor}_ID` adalah skor ID per-aktor; `ID_overall` hanyalah rata-ratanya. Jadi soalnya **sirkular**: model diminta menebak sesuatu yang sudah tertulis di dalam bahan yang ia pegang.

#### Kenapa itu masalah

> **Analogi:** Anda menyembunyikan angka gaji seseorang, tapi membiarkan kolom "gaji × 2" dan "gaji × 3" tetap terlihat. Siapa pun tinggal membagi dua. Model melakukan hal yang sama — bahkan tanpa "niat", karena algoritmanya memang mencari jalan terpendek ke jawaban.

Akurasi 85% yang keluar dari SET B **tidak membuktikan model mampu memprediksi apa pun**. Ia hanya membuktikan model bisa menghitung balik jawaban. Buktinya sudah tercetak di hasil lama File 06: tiga fitur yang paling diandalkan model persis ketiga fitur bocor itu (`IF_VfV_ID`, `IF_FCNC_ID`, `IF_VfV_FCNC`).

Kalau angka ini masuk Bab IV sebagai "kemampuan prediksi", penguji yang paham ML akan menemukannya — dan itu bisa meruntuhkan kredibilitas seluruh bagian ML.

#### Apa yang saya ubah

Satu himpunan "safe" diganti **tiga himpunan dengan label jujur** (File 05, sel 7):

| Himpunan | Isi | Boleh disebut |
|---|---|---|
| **SET A** | semua fitur, termasuk skor agregat (71 fitur) | demonstrasi kebocoran telanjang |
| **SET B** | skor agregat, `IF_*`, dan kolom kembar dibuang (50 fitur) | **pemeriksaan konsistensi internal** — BUKAN prediksi |
| **SET C** | hanya demografi/legal: usia, gender, pendidikan, jenis kasus, lama vonis, masa tahanan, lapas, wilayah (8 fitur) | **prediksi** — satu-satunya yang sah |

Ditambahkan juga **audit kebocoran yang berjalan otomatis**: sebelum model dilatih, kode mencoba sendiri menghitung balik skor konstruk dari fitur yang tersedia, lalu mencetak korelasinya. Ada `assert` yang **menggagalkan notebook** kalau SET C sampai tercemar.

```
=== AUDIT KEBOCORAN ===
  SET A : aljabar IF_*          korelasi=1.000000  <-- BOCOR
  SET A : rata2 ECV_{aktor}_ID  korelasi=1.000000  <-- BOCOR
  SET B : rata2 ECV_{aktor}_ID  korelasi=1.000000  <-- BOCOR
  SET C : tidak ada jalur rekonstruksi -> BERSIH
```

#### Justifikasi

Ada tiga pilihan yang dipertimbangkan:

| Pilihan | Kelebihan | Kekurangan | Keputusan |
|---|---|---|---|
| Hanya buang `IF_*` | perubahan paling kecil | **tidak menutup kebocoran** — jalur ECV tetap eksak | ditolak |
| Ganti target jadi forecasting antar-tahap | nol sirkularitas | mengubah pertanyaan penelitian, keluar dari proposal | ditolak |
| **Tiga himpunan berlabel jujur** | kebocoran diakui terbuka, analisis lama tetap terpakai, ada model prediksi bersih | perlu penjelasan lebih panjang di Bab IV | **dipilih** |

Kenapa SET B tetap dipertahankan meski bocor? Karena **pertanyaannya berbeda**. Bukan "bisakah kita memprediksi NREI" (tidak bisa — itu sirkular), melainkan *"seberapa koheren kategori NREI dengan pola jawaban granularnya, dan komponen granular mana yang paling menentukan"*. Itu pertanyaan yang sah, berguna, dan justru itulah yang membuat analisis SHAP di File 06 bermakna. Yang dilarang hanya **menyebutnya kemampuan prediktif**.

SET B dibiarkan bocor **secara terbuka dan tercatat** — auditnya mencetak korelasi 1.0 apa adanya. Menyembunyikan kebocoran jauh lebih berbahaya daripada mengakuinya sambil memberi label yang tepat.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Baseline kelas mayoritas | tidak dilaporkan | **0.543** |
| SET A | 0.920 ±0.024 *(70 fitur)* | 0.920 ±0.037 *(71 fitur)* |
| SET B | 0.804 ±0.057 — *diklaim "safe"* | 0.789 ±0.059 *(50 fitur)* — *dilabeli konsistensi internal* |
| Model prediksi bersih | **tidak ada** | **0.423 ±0.065** (SET C) |

> Catatan kejujuran: metode CV-nya juga berubah (lihat Bug 7), jadi angka SET A/B sebelum-sesudah tidak sepenuhnya apel-ke-apel. Yang penting bukan pergeseran kecil itu, melainkan **munculnya SET C**.

**Temuan paling penting:** SET C = **0.423**, sementara baseline = **0.543**. Model prediksi bersih **tidak mengalahkan tebakan buta**. Artinya: pada data ini, demografi tidak memprediksi persepsi governance sama sekali.

Untuk data dummy yang dibangkitkan acak, **itu hasil yang BENAR** — dan justru menjadi bukti bahwa pipeline-nya kini jujur. Kalau SET C tiba-tiba akurat pada data acak, itu tanda masih ada kebocoran tersisa.

**Cara melaporkan di Bab IV:**
- Angka SET B **jangan** disebut "akurasi prediksi". Sebut: *"kategori NREI dapat direkonstruksi dari pola jawaban granular dengan akurasi 0.789 — menunjukkan indeks komposit koheren dengan komponennya."*
- Angka SET C **boleh** disebut prediksi. Kalau tidak mengalahkan baseline, laporkan apa adanya: itu temuan negatif yang sah, bukan kegagalan metode.

---

### Bug 2 — Model yang dianalisis bukan model juara

#### Apa yang salah

File 05 membandingkan tiga model, menyimpulkan juaranya, lalu menyimpan model itu untuk dianalisis di File 06. Tapi baris penyimpanannya salah tulis:

```python
best_model_obj = models[best_model_name] if ... else rf   # dihitung...
joblib.dump({"model": rf, ...})                           # ...lalu tidak dipakai!
```

Variabel `best_model_obj` dihitung tapi dibuang. Yang benar-benar disimpan adalah `rf` — Random Forest — secara *hardcoded*, padahal yang dilaporkan sebagai juara adalah XGBoost.

#### Kenapa itu masalah

> **Analogi:** Panitia mengumumkan Juara 1 adalah peserta A, tapi yang difoto, diwawancara, dan dibedah gaya bermainnya di koran adalah peserta B.

Seluruh penjelasan "kenapa model memutuskan begini" di File 06 (SHAP dan LIME) sebenarnya menjelaskan **model yang salah**. Kesimpulan Bab IV tentang "fitur mana yang paling berpengaruh" jadi tidak nyambung dengan model yang diklaim dipakai.

#### Apa yang saya ubah

File 05 sel 20 — variabel yang benar dipakai, plus identitas model ikut dicatat supaya bisa diaudit:

```python
joblib.dump({"model": best_model,
             "model_name": best_model_name,   # baru: bisa dicocokkan
             "X_train": Xtr,                  # baru: untuk LIME (Bug 5)
             "y_train": ytr,
             ...
             "feature_set": "SET B (konsistensi internal, BUKAN prediksi)"},
            "classifier_for_shap.joblib")
```

File 06 juga disesuaikan agar tidak lagi berasumsi modelnya pasti berbasis pohon: kalau juaranya Logistic Regression, SHAP otomatis memakai `LinearExplainer` alih-alih `TreeExplainer` (kalau tidak, notebook akan error).

#### Justifikasi

Ini murni salah ketik, jadi perbaikannya cukup satu baris. Tambahan `model_name` dan `feature_set` dimasukkan karena tanpa itu **bug yang sama bisa terulang tanpa terdeteksi** — sekarang ada tes otomatis yang mencocokkan nama model di bundle dengan juara di `model_comparison.csv`.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Juara yang dilaporkan | XGBoost | Random Forest |
| Model yang disimpan | Random Forest ❌ **tidak cocok** | Random Forest ✅ **cocok** |
| Tes otomatis | tidak ada | `tests/test_bundle.py` — **LOLOS** |

---

### Bug 3 — `L_total` menjumlahkan angka dari dua model berbeda

#### Apa yang salah

Rumus `L_total = L_classification + λ₁·L_fairness + λ₂·L_ethical` dihitung dari sumber yang tidak konsisten:

- `L_classification` dari probabilitas **Random Forest**
- `L_fairness` dari kesalahan prediksi **XGBoost**

Lalu keduanya dijumlahkan seolah berasal dari satu model.

#### Kenapa itu masalah

> **Analogi:** Menjumlahkan nilai matematika si A dengan nilai bahasa si B, lalu menyebutnya rapor satu orang.

Angka `L_total` jadi tidak punya makna — ia tidak menggambarkan performa model mana pun.

#### Apa yang saya ubah

File 05 sel 13 — semua komponen dihitung dari satu objek model yang sama (`best_model`), dengan penanganan scaling bila modelnya Logistic Regression. Ditambah pencetakan eksplisit nama modelnya supaya terlihat.

Ditambahkan juga peringatan kejujuran: jumlah anggota tiap subgroup di data test dicetak (6–11 orang), dengan catatan bahwa `L_fairness` pada ukuran itu **indikatif saja, belum bisa diklaim signifikan**.

#### Justifikasi

Perbaikan minimal dan tidak ada pilihan lain — komponen sebuah rumus tunggal wajib berasal dari model tunggal. Catatan soal ukuran subgroup ditambahkan karena tanpa itu pembaca bisa memperlakukan `L_fairness` sebagai temuan kokoh, padahal dengan 6 orang per kelompok satu kesalahan saja menggeser angkanya drastis.

#### Angka sebelum → sesudah

| Komponen | Sebelum | Sesudah |
|---|---|---|
| `L_classification` | 0.4378 *(dari RF)* | 0.4863 *(dari RF)* |
| `L_fairness` | 0.0155 *(dari XGBoost)* ❌ | 0.0053 *(dari RF)* ✅ |
| `L_ethical` | 0.5138 | 0.5138 |
| **`L_total`** | **0.5421** *(campuran)* | **0.5895** *(satu model)* |

---

### Bug 4 — Mengisi data kosong dengan cara yang "mengintip" data ujian

#### Apa yang salah

Sebagian responden punya data kosong. Kode lama mengisinya dengan rata-rata:

```python
X_safe = X_safe.fillna(X_safe.mean())     # <-- SELURUH data, sebelum dibagi
Xtr, Xte, ... = train_test_split(X_safe, ...)
cv = cross_val_score(rf, X, y, cv=...)    # <-- CV pun memakai data yang sudah tercampur
```

Rata-rata penggantinya dihitung dari **semua** data — termasuk data ujian yang seharusnya disembunyikan — dan dilakukan **sebelum** pembagian.

#### Kenapa itu masalah

> **Analogi:** Sebelum ujian dimulai, guru merangkum sedikit informasi dari soal ujian dan menyelipkannya ke buku latihan murid. Kecil, tapi tetap bocoran.

Nilai ujian model jadi sedikit lebih bagus dari yang seharusnya. Efeknya kecil di sini, tapi ini menyalahi kaidah baku dan penguji yang teliti bisa mempersoalkannya.

#### Apa yang saya ubah

Imputasi dipindahkan **ke dalam `Pipeline` scikit-learn** (File 05 sel 7), sehingga otomatis di-`fit` hanya pada bagian training di setiap lipatan CV:

```python
def buat_pipe(model, scale=False, k_select=None):
    steps = [("imputer", SimpleImputer(strategy="mean"))]
    if scale:      steps.append(("scaler", StandardScaler()))
    if k_select:   steps.append(("select", SelectKBest(f_classif, k=k_select)))
    steps.append(("model", model))
    return Pipeline(steps)
```

Untuk split tunggal (sel 9), imputer di-`fit` eksplisit hanya pada `Xtr_raw` lalu diterapkan ke `Xte_raw` — bukan sebaliknya.

#### Justifikasi

`Pipeline` adalah mekanisme baku scikit-learn justru untuk mencegah kebocoran semacam ini. Menghitung imputasi manual per-lipatan bisa saja, tapi rawan salah dan sulit diaudit. Bonusnya: pola yang sama otomatis melindungi *scaling* dan *seleksi fitur* yang ditambahkan kemudian (Bug 12) — kalau seleksi fitur dilakukan di luar CV, itu bentuk kebocoran yang lain lagi.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Kapan imputasi dihitung | sebelum split, dari seluruh data | di dalam Pipeline, per lipatan training |
| Cakupan kebocoran | statistik test merembes ke training | tidak ada |
| Tes otomatis | tidak ada | `test_bundle.py` — 0 NaN, train/test **0 baris beririsan** |

---

### Bug 5 — LIME dikalibrasi dengan data yang salah

#### Apa yang salah

```python
lime_explainer = LimeTabularExplainer(training_data=np.array(X_test), ...)
```

Parameternya bernama `training_data`, tapi yang diberikan adalah **X_test** (40 baris).

#### Kenapa itu masalah

LIME memakai data ini untuk menghitung statistik perturbasi dan batas diskretisasi — jadi seharusnya data **training** (159 baris). Dengan 40 baris, penjelasan lokalnya tidak stabil, dan diam-diam ia mengambil informasi dari data uji.

#### Apa yang saya ubah

File 05 kini menyimpan `X_train` di bundle; File 06 sel 6 memakainya:

```python
lime_explainer = LimeTabularExplainer(training_data=np.array(X_train), ...)
print(f"LIME dikalibrasi dgn {len(X_train)} baris data TRAINING "
      f"(sebelumnya: {len(X_test)} baris data test)")
```

#### Justifikasi

Perbaikan satu baris, tidak ada trade-off. Baris cetak ditambahkan agar setiap kali notebook dijalankan, angkanya terlihat — sehingga kalau bug ini terulang, langsung kentara.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Data kalibrasi LIME | 40 baris (data test) ❌ | **159 baris (data training)** ✅ |

---

### Bug 6 — Uji Sobol membandingkan dengan beban yang tidak sama

#### Apa yang salah

File 03 menguji: kalau bobot rumus digoyang sedikit, apakah kesimpulannya berubah? Parameternya didefinisikan begini:

```python
"bounds": [[0.50,0.70],   # w_mas1   -> bergeser ±0.10
           [0.40,0.60],   # w_eci1   -> bergeser ±0.10
           [0.30,0.50],   # w_nrei1  -> bergeser ±0.10
           [0.30,0.50]]   # w_nrei2  -> bergeser ±0.10
w_nrei3 = 1 - w_nrei1 - w_nrei2   # <-- bobot GR: ikut bergeser 0.40 s/d 0!
```

Karena `w_nrei1` dan `w_nrei2` digoyang **independen**, bobot GR yang dihitung sebagai sisanya bergeser dari 0.40 sampai 0 — yaitu **±0.20**, dua kali lipat yang lain. Total "anggaran goyangan" di level NREI menjadi 0.40, sedangkan level MAS dan ECI hanya 0.20.

#### Kenapa itu masalah

> **Analogi:** Menguji dua jembatan — yang satu dibebani 1 ton, yang satu 5 ton — lalu menyimpulkan jembatan kedua "lebih lemah" karena lebih bergetar.

Kesimpulan lama berbunyi: *"bobot NREI paling bertanggung jawab atas kerapuhan (ST > 0.80), sedangkan MAS/ECI sangat stabil (ST < 0.03)."* Kesimpulan itu **terkonfound oleh desain** — level NREI memang diberi ruang goyang lebih lebar, jadi wajar kalau tampak paling berpengaruh.

#### Apa yang saya ubah

Karena bobot harus berjumlah 1, **mustahil** memberi setiap bobot rentang independen yang identik. Yang bisa disetarakan adalah **total pergeseran** tiap level, yaitu Σ|Δw| ≤ B dengan B sama untuk semua level (File 03 sel 10):

| Level | Parametrisasi baru | Σ\|Δw\| maksimum |
|---|---|---|
| MAS (2 bobot) | `w = (0.60+d, 0.40−d)`, d ∈ [−B/2, +B/2] | 2·(B/2) = **B** |
| ECI (2 bobot) | `w = (0.50+d, 0.50−d)`, d ∈ [−B/2, +B/2] | 2·(B/2) = **B** |
| NREI (3 bobot) | `w = (0.40+a, 0.40+b, 0.20−a−b)`, a,b ∈ [−B/4, +B/4] | ≤ 2(\|a\|+\|b\|) = **B** |

Dengan **B = 0.20**, rentang `w_mas1` dan `w_eci1` **tetap persis seperti versi lama** — yang berubah hanya level NREI yang tadinya kelebihan beban.

Ditambah dua hal: kode **mencetak cek anggaran** (agar ketimpangan langsung terlihat kalau terulang), dan **menjalankan kedua versi berdampingan** agar efek koreksinya terukur, bukan cuma diklaim.

```
Cek anggaran deviasi L1 tiap level:
  LAMA: L1_MAS=0.20  L1_ECI=0.20  L1_NREI=0.40   -> TIMPANG
  BARU: L1_MAS=0.20  L1_ECI=0.20  L1_NREI=0.20   -> SETARA
```

#### Justifikasi

Pilihan "anggaran L1 yang sama" dipilih karena ia satu-satunya yang **bisa dipertahankan secara matematis** di bawah kendala jumlah bobot = 1. Alternatif "beri semua bobot rentang ±0.10 yang sama" terdengar lebih intuitif tapi **tidak mungkin** — begitu dua bobot digeser, yang ketiga terpaksa menanggung selisihnya.

Kedua versi dijalankan bersama karena ini bukan sekadar perbaikan teknis: kesimpulan di Bab IV berubah besarannya, dan pembaca berhak melihat berapa banyak yang berubah.

#### Angka sebelum → sesudah

| Parameter | ST sebelum | ST sesudah |
|---|---|---|
| `w_mas1` | 0.0322 | **0.1111** |
| `w_eci1` | 0.0159 | **0.0596** |
| `w_nrei1` | 0.8416 | 0.7155 |
| `w_nrei2` | 0.7917 | 0.6410 |
| **Rasio ST(NREI) / ST(MAS+ECI)** | **34.0×** | **7.94×** |

**Cara membacanya:** arah kesimpulan lama **tetap benar** — bobot NREI memang paling berpengaruh. Tapi besarannya dulu **berlebih sekitar 4 kali**. Dan MAS/ECI ternyata **tidak** "nyaris nol" seperti yang dilaporkan sebelumnya.

Ini kabar baik untuk disertasi: temuan utamanya bertahan, hanya perlu dinyatakan dengan angka yang benar.

---

## Bagian II — Robustness statistik

---

### Bug 7 — "Ujian" modelnya hanya diikuti 40 orang

#### Apa yang salah

Semua angka akurasi berasal dari **satu kali** pembagian data, dengan hanya 40 orang di bagian ujian. Selisih antar model (0.825 vs 0.875) setara **1 orang**.

#### Kenapa itu masalah

> **Analogi:** Menyimpulkan calon A unggul dari calon B lewat survei ke 40 orang, padahal selisihnya 1 suara. Ulangi dengan 40 orang lain, hasilnya bisa terbalik.

Dengan 40 orang, "akurasi 87,5%" sebenarnya bisa saja 78% atau 95%.

#### Apa yang saya ubah

Angka headline sekarang dari **repeated stratified cross-validation** — 5 lipatan × 10 ulangan = **50 evaluasi** — dan dilaporkan **beserta rentangnya**:

```
SET B (konsistensi internal)  akurasi = 0.789 +/- 0.059  [p2.5-p97.5: 0.675-0.875]
```

Split tunggal tetap dipertahankan, tapi **hanya** untuk menghasilkan satu model terlatih (untuk confusion matrix dan SHAP), dengan peringatan yang dihitung otomatis:

```
CATATAN: n_test=40, jadi 1 sampel = 0.025 akurasi. Selisih antar
model di bawah 0.050 TIDAK bermakna -- jangan diklaim satu model unggul.
```

#### Justifikasi

Dengan 199 responden, membuang 40 orang untuk ujian tunggal adalah pemborosan informasi. Repeated CV memakai **setiap** responden sebagai data uji berkali-kali, sehingga estimasinya jauh lebih stabil, sekaligus memberi **rentang** — dan rentang inilah yang mencegah klaim berlebihan.

Split tunggal tidak dihapus karena SHAP/LIME memang butuh satu model konkret yang terlatih.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Metode | 1 split + 5-fold CV | **repeated stratified CV 5×10 = 50 evaluasi** |
| Pelaporan | angka tunggal | rata-rata **+ simpangan + rentang p2.5–p97.5** |
| Baseline pembanding | tidak ada | **0.543** |
| Peringatan ukuran sampel | tidak ada | dihitung & dicetak otomatis |

---

### Bug 8 — XGBoost diadu tanpa penyeimbang kelas

#### Apa yang salah

Dua dari tiga model diberi `class_weight="balanced"`, XGBoost tidak:

```python
"Random Forest":       RandomForestClassifier(..., class_weight="balanced"),
"XGBoost":             xgb.XGBClassifier(...),                    # <-- tanpa penyeimbang
"Logistic Regression": LogisticRegression(..., class_weight="balanced"),
```

#### Kenapa itu masalah

Distribusi target tidak seimbang (Moderate 108, Low 50, High 41). Model yang diseimbangkan dan yang tidak, diperlakukan berbeda — jadi perbandingannya tidak apel-ke-apel, dan penobatan "model terbaik" tidak sah.

#### Apa yang saya ubah

XGBoost tidak punya parameter `class_weight`, jadi penyetaraannya lewat `sample_weight`:

```python
w_tr = compute_sample_weight("balanced", ytr)
...
model.fit(Xtr, ytr, sample_weight=w_tr)   # setara "balanced" pada RF/LogReg
```

#### Justifikasi

`compute_sample_weight("balanced", y)` adalah utilitas resmi scikit-learn yang menghasilkan bobot **identik** dengan yang dipakai `class_weight="balanced"`. Jadi ketiga model kini benar-benar diperlakukan sama.

#### Angka sebelum → sesudah

| Model | macro-F1 sebelum | macro-F1 sesudah |
|---|---|---|
| Random Forest | 0.8253 | **0.8819** ← juara |
| XGBoost | 0.8740 ← juara | 0.8326 |
| Logistic Regression | 0.7934 | 0.8247 |

**Konsekuensi penting:** juaranya berubah **XGBoost → Random Forest**. Ini bukan bug baru, tapi akibat langsung dari perbandingan yang kini adil. Angka di Bab IV perlu diperbarui.

---

### Bug 9 — Skor kestabilan cluster diabaikan oleh notebook sendiri

#### Apa yang salah

Kode menghitung Adjusted Rand Index (ARI) antara K-Means dan Hierarchical, mencetak keterangan *"makin tinggi makin yakin cluster-nya nyata"*, mendapat hasil **0.184** — lalu **melanjutkan seolah 2 cluster itu valid**.

#### Kenapa itu masalah

> **Analogi:** Dua juri diminta membagi peserta jadi dua kelompok bakat. Kalau pembagian juri 1 dan juri 2 hampir tidak ada miripnya, kita patut curiga kelompok itu tidak benar-benar ada.

ARI 0.184 dari maksimal 1.0 berarti kedua algoritma **hampir tidak sepakat**. Kelompok yang tidak stabil tidak layak diceritakan sebagai "tipologi responden" di Bab IV.

#### Apa yang saya ubah

Ditambahkan ambang dan putusan eksplisit (File 05 sel 4):

```
-> ARI < 0.5: kedua algoritma TIDAK sepakat soal keanggotaan cluster.
   Cluster ini dinilai TIDAK STABIL. Profil di bawah tetap dicetak sebagai
   deskripsi, TAPI tidak boleh dilaporkan sebagai tipologi/temuan di Bab IV.
```

Statusnya juga direkam ke `objective_report.csv` sebagai kolom `cluster_stabil` (nilainya: `False`), supaya tidak bisa "hilang" saat penulisan laporan.

#### Justifikasi

Profil cluster tetap dicetak, bukan dihapus — ia masih berguna sebagai deskripsi. Yang ditambahkan adalah **rambu** agar tidak naik kelas menjadi temuan. Statusnya ditulis ke CSV karena peringatan yang hanya tercetak di layar mudah terlewat berbulan-bulan kemudian saat menulis Bab IV.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| ARI | 0.184 | 0.184 *(tidak berubah — ini fakta data)* |
| Putusan kestabilan | tidak ada, narasi lanjut seolah valid | **"TIDAK STABIL"** eksplisit |
| Terekam di CSV | tidak | `cluster_stabil = False` |

---

### Bug 10 — "Ditemukan 10 anomali", padahal angka 10 itu kita yang menyetel

#### Apa yang salah

```python
iso = IsolationForest(contamination=0.05, random_state=42)
print(f"Isolation Forest: {df['anomaly_iso'].sum()} anomali")   # selalu 10
```

`contamination=0.05` berarti perintah *"tandai 5% data sebagai anomali"* — apa pun isinya. Dari 199 responden, 5% ≈ 10 orang. Jadi "ditemukan 10 anomali" bukan penemuan; itu perintah kita sendiri yang dikembalikan.

#### Kenapa itu masalah

> **Analogi:** Menyuruh satpam "apa pun yang terjadi, laporkan 10 orang paling mencurigakan setiap malam" — lalu heran kenapa setiap malam selalu ada tepat 10 orang mencurigakan.

Di data riil ini berbahaya: 10 orang akan selalu tertandai meskipun sebenarnya tidak ada satu pun yang aneh.

#### Apa yang saya ubah

Hasil utama kini **skor keanehan kontinu**, bukan cap anomali:

```python
df["anomaly_score_iso"] = -iso.score_samples(Xa_scaled)   # makin tinggi = makin tidak wajar
df["anomaly_score_lof"] = -lof.negative_outlier_factor_
```

Jumlah yang tertandai tetap dicetak, tapi dengan label yang jujur:

```
=== JUMLAH TERTANDAI PADA AMBANG contamination=0.05 ===
Isolation Forest : 10 ditandai  <- konsekuensi parameter, BUKAN temuan
Z-score (|z|>3)  : 0 anomali    (ambang statistik, bukan kuota)
```

Ditambah daftar **5 responden dengan skor tertinggi** sebagai prioritas review manusia — yang secara praktis lebih berguna daripada daftar biner.

#### Justifikasi

Metodenya sendiri tidak salah; yang salah adalah **cara melaporkannya**. Menghapus `contamination` bukan pilihan karena Isolation Forest memerlukannya. Solusinya: pindahkan penekanan ke skor kontinu (yang memang tidak bergantung ambang), dan jadikan penetapan ambang sebagai keputusan manusia yang eksplisit — ini juga selaras dengan prinsip *Human-in-the-Loop* di proposal.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Hasil utama | "10 anomali ditemukan" | **skor keanehan kontinu** (+ statistik sebarannya) |
| Kolom baru di CSV | — | `anomaly_score_iso`, `anomaly_score_lof` |
| Framing jumlah | temuan | "konsekuensi parameter, BUKAN temuan" |

---

## Bagian III — Metodologi

---

### Bug 11 — Rotasi EFA yang mengasumsikan hal yang tidak benar

#### Apa yang salah

Tiga kelemahan sekaligus di File 01:

1. **Rotasi varimax** mengasumsikan ketiga konstruk VfV/FCNC/ID **tidak saling berkorelasi** — padahal secara teori (dan kenyataan) mereka berkaitan.
2. **Hanya kriteria Kaiser** (eigenvalue > 1) untuk menentukan jumlah faktor — kriteria yang dikenal cenderung melebih-lebihkan.
3. **Non-independensi observasi** hanya dicatat sebagai keterbatasan: 896 baris berasal dari ~200 orang (satu orang menilai sampai 5 aktor), jadi uji statistiknya memperlakukan 896 baris sebagai 896 orang independen.

#### Kenapa itu masalah

> **Analogi (poin 1):** Memaksa peta dunia digambar dengan asumsi Bumi itu datar. Petanya tetap bisa dibaca, tapi bentuk benua di pinggir jadi melenceng — dan Anda tidak tahu mana yang melenceng karena asumsi, mana yang memang bentuknya.
>
> **Analogi (poin 3):** Mengklaim "896 orang setuju" padahal yang disurvei 200 orang, masing-masing ditanya hingga 5 kali.

Akibat konkret dari varimax: cross-loading 0.22–0.30 yang muncul di hasil lama sebagian hanyalah **artefak rotasi**, bukan sifat itemnya — sehingga item bisa dicurigai ambigu padahal sebetulnya tidak.

#### Apa yang saya ubah

Tiga perbaikan, semuanya di File 01:

**(1) Rotasi promax (oblique)** menggantikan varimax. Bonusnya: korelasi antar-faktor kini bisa dibaca langsung — informasi yang sama sekali tidak tersedia pada varimax.

**(2) Parallel analysis (Horn, 1965)** dilaporkan berdampingan dengan Kaiser. Caranya: bangkitkan 500 set data acak berukuran sama, hitung eigenvalue-nya, ambil persentil ke-95. Faktor dipertahankan hanya kalau eigenvalue nyatanya **melampaui** apa yang bisa muncul dari data acak belaka.

```
 faktor   ev_nyata  ambang_acak  keputusan
      1      8.474        1.335  TAHAN
      2      1.645        1.261  TAHAN
      3      1.419        1.207  TAHAN
      4      0.360        1.163  buang
-> Kedua kriteria SEPAKAT: 3 faktor. Kesimpulan kokoh.
```

**(3) EFA pembanding pada satu aktor saja** — hanya penilaian terhadap Investigator, di mana setiap orang tepat satu baris, sehingga observasinya **independen penuh**.

#### Justifikasi

Untuk konstruk psikososial yang secara teori berkaitan, rotasi oblique adalah standar; varimax hanya tepat bila faktor benar-benar diasumsikan ortogonal. Dan asumsi itu **bisa diuji** — itulah gunanya tabel korelasi antar-faktor.

Parallel analysis tidak menggantikan Kaiser tapi **dilaporkan bersamanya**, karena kesepakatan dua kriteria jauh lebih kuat di sidang daripada satu kriteria yang diketahui lemah.

EFA pembanding dipilih daripada model multilevel karena ia **menjawab keberatan dengan data, bukan dengan asumsi tambahan** — dan jauh lebih mudah dijelaskan di sidang: *"kalau strukturnya tetap muncul pada responden yang benar-benar independen, keberatan itu terjawab."*

#### Angka sebelum → sesudah

| | Sebelum (varimax) | Sesudah (promax) |
|---|---|---|
| Loading utama | 0.72 – 0.78 | **0.79 – 0.86** |
| Cross-loading | 0.22 – 0.30 | **0.00 – 0.06** |
| Korelasi antar-faktor | tidak terlihat | **0.597 – 0.644** |
| Kriteria jumlah faktor | Kaiser saja → 3 | Kaiser **dan** parallel analysis → **3 (sepakat)** |
| Uji non-independensi | catatan kaki | **EFA N=129 independen: 3 faktor, 15/15 item ≥0.50, KMO 0.931** |

**Ini hasil paling dramatis dari seluruh perbaikan.** Cross-loading runtuh dari 0.22–0.30 menjadi hampir nol — membuktikan kekhawatiran tentang item ambigu memang **artefak rotasi**, persis seperti dugaan di laporan review.

Dan korelasi antar-faktor **0.60–0.64** bukan angka kecil: itu bukti empiris bahwa VfV/FCNC/ID memang konstruk yang saling berkaitan erat. Temuan substantif yang layak dilaporkan — dan yang selama ini **tersembunyi** karena varimax memaksanya menjadi nol.

---

### Bug 12 — Terlalu banyak fitur, sebagian kembar identik

#### Apa yang salah

Model diberi 58–70 fitur untuk 199 responden. Lebih buruk lagi, sebagian fitur adalah **salinan persis** fitur lain dengan nama berbeda:

```
VfV_Investigator        vs  ECV_Investigator_VfV        max_diff=0.0  identik=True
VfV_Prosecutor          vs  ECV_Prosecutor_VfV          max_diff=0.0  identik=True
VfV_Judge               vs  ECV_Judge_VfV               max_diff=0.0  identik=True
VfV_Advocate            vs  ECV_Advocate_VfV            max_diff=0.0  identik=True
VfV_CorrectionalOfficer vs  ECV_CorrectionalOfficer_VfV max_diff=0.0  identik=True
```

#### Kenapa itu masalah

> **Analogi:** Menyuruh detektif menyimpulkan pola dari 199 kasus, tapi membekalinya 70 jenis petunjuk — sebagian petunjuk malah fotokopian petunjuk lain. Makin banyak petunjuk untuk kasus sesedikit itu, makin mudah ia "menemukan" pola yang cuma kebetulan.

Dua dampak: (1) inilah pintu masuk **overfitting** sesungguhnya, terutama nanti di data riil; (2) fitur kembar **memecah nilai SHAP** — kepentingan satu informasi terbagi ke beberapa kolom salinannya, sehingga peringkat fitur tidak bisa dibaca apa adanya.

#### Apa yang saya ubah

**(1) Kolom kembar dikeluarkan** dari fitur model (File 05 sel 7) — dengan `assert` agar tidak bisa masuk lagi diam-diam.

**(2) Rasio fitur:responden dicetak** supaya selalu terpantau:

```
Rasio fitur:responden SET B = 50:199 = 1:4.0  (makin kecil makin rawan overfitting)
```

**(3) Cek overfitting empiris** dengan seleksi fitur **di dalam** Pipeline:

```python
s_b20 = evaluasi("SET B + SelectKBest(k=20)", X_b, k_select=20)
s_b10 = evaluasi("SET B + SelectKBest(k=10)", X_b, k_select=10)
```

#### Justifikasi

Kolom kembar dibuang dari **fitur model**, bukan dari `feature_matrix.csv`. Alasannya: kedua blok kolom itu (pivot `IF_VfV,Actor` dan `ECV`) **diminta eksplisit oleh proposal** (Tabel 3.28 dan 3.31), jadi menghapusnya dari berkas keluaran akan merusak deliverable. Yang perlu dicegah hanyalah memakai keduanya sekaligus sebagai prediktor.

`SelectKBest` ditempatkan **di dalam** `Pipeline` — bukan sebelum CV — karena menyeleksi fitur dengan melihat seluruh data adalah bentuk kebocoran yang lain lagi (fitur "terbaik" dipilih dengan bantuan data uji).

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Fitur SET B | 58 | **50** *(5 kembar + 4 `IF_*` dibuang, 1 bersih ditambah)* |
| Rasio fitur:responden | tidak dilaporkan | **1:4.0** dicetak otomatis |
| Uji overfitting | tidak ada | k=20 → **0.768**; k=10 → **0.736**; 50 fitur → **0.789** |

**Cara membacanya dengan jujur:** memangkas fitur **menurunkan** akurasi (0.789 → 0.768 → 0.736). Jadi fitur tambahan itu memang membawa informasi, bukan sekadar sampah — dan notebook **melaporkan itu apa adanya**, tidak dipaksa menyimpulkan "fitur sedikit sama baiknya".

Untuk data riil nanti, angka-angka ini jadi dasar keputusan: kalau selisihnya kecil, versi terpangkas lebih aman dari overfitting.

---

### Bug 13 — Pemenang bobot ERM dipilih dari selisih yang nyaris nol

#### Apa yang salah

Empat skema bobot ERM dibandingkan, lalu yang standar deviasinya tertinggi dinobatkan pemenang:

```
procedural_emphasis  std = 0.323276   <-- "menang"
integrity_emphasis   std = 0.322993
value_emphasis       std = 0.322540
equal                std = 0.321678
```

Selisih juara dan juru kunci: **0.0016** — di angka desimal ketiga.

#### Kenapa itu masalah

> **Analogi:** Empat pelari finis dengan selisih seperseribu detik, diukur dengan stopwatch tangan yang ketelitiannya sepersepuluh detik — lalu satu dinobatkan juara.

"Pemenang" itu ditentukan oleh kebetulan data, bukan perbedaan nyata. Kalau data dummy dibangkitkan ulang dengan seed lain, pemenangnya bisa berganti.

#### Apa yang saya ubah

Ditambahkan **uji kelayakan sebelum menobatkan** (File 06 sel 10): selisih harus lebih dari 1% dari std tertinggi untuk dianggap nyata.

```
Rentang std antar skema: 0.001597 (0.494% dari std tertinggi)
Ambang kelayakan: selisih harus > 1% dari std tertinggi
-> SERI: keempat skema TIDAK dapat dibedakan pada data ini.
   Menobatkan 'procedural_emphasis' sebagai pemenang berarti memilih berdasarkan NOISE.
   Dipakai skema netral 'equal' sebagai default, dan yang DILAPORKAN di Bab IV
   adalah: keempat skema setara -> penentuan bobot MENUNGGU expert panel.
```

#### Justifikasi

Notebook lama sudah benar melabeli ini *placeholder* menunggu expert panel — niatnya sudah tepat. Yang kurang: ia tetap **menobatkan satu nama**, dan nama itu berisiko dikutip di Bab IV seolah hasil analisis.

Saat seri, dipakai `equal` sebagai default karena ia **netral** — tidak memihak konstruk mana pun, sehingga tidak mendahului keputusan expert panel. Memakai `procedural_emphasis` (juara semu) justru menanam bias yang tak berdasar.

Ambang 1% dipilih sebagai batas konservatif: jauh di atas noise sampling, tapi masih cukup peka menangkap perbedaan nyata kalau memang ada di data riil.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Skema terpilih | `procedural_emphasis` *(dari selisih 0.494%)* | `equal` *(netral, karena **SERI**)* |
| Uji kelayakan selisih | tidak ada | ada — 0.494% < ambang 1% |
| Status dilaporkan | "skema paling diskriminatif" | **"keempat skema setara, menunggu expert panel"** |

---

### Bug 14 — Flag "asal centang" dibuat lalu dilupakan

#### Apa yang salah

File 00 mendeteksi responden yang menjawab lurus semua (*straight-lining*) dan menandainya di kolom `flag_straightline`. Tapi kolom itu **tidak pernah dipakai** di File 01–07. Ke-11 responden itu tetap ikut semua analisis tanpa catatan apa pun.

#### Kenapa itu masalah

Deteksi tanpa tindak lanjut sama dengan tidak mendeteksi. Penguji yang melihat ada mekanisme deteksi akan bertanya: *"lalu diapakan?"* — dan jawaban "tidak diapa-apakan" sulit dipertahankan.

#### Apa yang saya ubah

Flag-nya kini **dikonsumsi sebagai sensitivity check** (File 05): analisis diulang tanpa mereka, lalu selisihnya dilaporkan.

```
=== SENSITIVITY CHECK: straight-lining (11 responden) ===
SET B tanpa straight-liner   akurasi = 0.786 +/- 0.060
Selisih akurasi tanpa straight-liner: -0.003
-> Dampaknya kecil (<0.03): kesimpulan TIDAK bergantung pada responden ini.
   Mereka boleh dipertahankan, dan fakta ini yang dilaporkan di Bab IV.
```

#### Justifikasi

Dipilih *sensitivity check* daripada membuang data, karena:

1. **Membuang data adalah keputusan tak terbalikkan** yang seharusnya di tangan peneliti, bukan skrip.
2. Straight-lining **belum tentu** berarti tidak serius — bisa juga responden yang memang konsisten menilai semua aspek sama rendah/tinggi.
3. Sensitivity check **menjawab pertanyaan yang lebih baik**: bukan "apakah mereka valid?" (tidak bisa dijawab dari data), melainkan "apakah kesimpulan kita bergantung pada mereka?" — dan itu **bisa** dijawab, dengan angka.

Flag diambil dari `data_index_actor.csv` yang sudah memuatnya, jadi File 00 dan File 04 tidak perlu diubah sama sekali.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Flag dipakai | tidak, sama sekali | ya — sensitivity check |
| Responden terdampak | 11 (tak diperiksa) | 11 (diukur dampaknya) |
| Dampak pada kesimpulan | tidak diketahui | **−0.003** → dapat diabaikan |

---

### Bug 15 — Bobot yang sesungguhnya dominan tidak pernah disebut

#### Apa yang salah

Rumus NREI terlihat memberi bobot seimbang: `0.40×MAS + 0.40×ECI + 0.20×GR`. Tapi karena **ID masuk ke MAS dan ECI sekaligus**, bobot sesungguhnya pada tingkat konstruk dasar sangat berbeda — dan itu tidak pernah ditulis.

#### Kenapa itu masalah

Ini bukan kesalahan hitung, tapi konsekuensi matematis yang tidak terlihat dari rumusnya. Tiga alasan ia wajib ditulis:

1. Ia **menjelaskan banyak hasil hilir** — misalnya kenapa fitur yang mengandung ID selalu muncul teratas di SHAP.
2. Kalau tidak disebut, penguji yang menghitungnya sendiri akan menyimpulkan penulis **tidak menyadarinya**.
3. Ia relevan untuk sensitivity analysis: pergeseran bobot yang menyentuh ID berdampak lebih besar.

#### Apa yang saya ubah

Ditambahkan tabel penjabaran di File 02 (markdown, tanpa mengubah perhitungan apa pun):

| Konstruk | Jalur | Bobot efektif di NREI |
|---|---|---|
| **ID** | 0.40×0.40 (via MAS) + 0.40×0.50 (via ECI) | **0.36** ← terbesar |
| VfV | 0.40 × 0.60 (via MAS) | 0.24 |
| FCNC | 0.40 × 0.50 (via ECI) | 0.20 |
| GR | langsung | 0.20 |
| | | **jumlah = 1.00** ✓ |

#### Justifikasi

**Tidak ada kode yang diubah** — dan itu memang disengaja. Bobot ini berasal dari desain proposal, di mana ID (Integrity & Discipline) memang diposisikan sebagai fondasi dua indeks sekaligus. Mengubahnya berarti mengubah kerangka teori, yang jauh di luar wewenang perbaikan bug.

Yang dibutuhkan hanyalah **transparansi**: ubah dari "tidak disadari" menjadi "disadari dan dijelaskan".

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Bobot ID | 0.36 *(tidak didokumentasikan)* | 0.36 *(didokumentasikan lengkap dengan turunannya)* |
| Perhitungan | tidak diubah | tidak diubah |

---

## Bagian IV — Temuan baru selama perbaikan

Dua hal berikut **tidak ada di laporan review** — keduanya baru terlihat setelah kode dijalankan di lingkungan ini. Keduanya sudah dikonfirmasi ke Anda sebelum diperbaiki.

---

### Temuan (a) — Notebook mencetak klaim yang bertentangan dengan angkanya sendiri

#### Apa yang salah

```python
print(f"L_fairness sesudah weighting: {L_fairness_w:.4f}  (turun = lebih adil antar subgroup)")
```

Label "(turun = lebih adil)" dicetak **tanpa syarat**, apa pun angkanya. Di Colab dulu angkanya kebetulan turun (0.0155 → 0.0099), jadi tidak kelihatan. Saat dijalankan di lingkungan ini, angkanya **naik** (0.0053 → 0.0132) — tapi notebook tetap berbunyi "turun".

#### Kenapa itu masalah

Notebook mencetak kesimpulan yang **bertentangan dengan data yang baru saja ia hitung sendiri**. Kalau kalimat ini dikutip ke Bab IV, isinya salah — dan kesalahannya tidak akan terdeteksi karena angka dan narasi dicetak berdampingan seolah saling mendukung.

Ini juga menunjukkan hal yang lebih umum: *fairness weighting* **tidak dijamin** memperbaiki `L_fairness`, terutama pada subgroup kecil.

#### Apa yang saya ubah

Arah perubahan kini dibaca dari data:

```python
delta = L_fairness_w - L_fairness
arah = "TURUN" if delta < 0 else ("NAIK" if delta > 0 else "TIDAK BERUBAH")
print(f"L_fairness sesudah weighting: {L_fairness_w:.4f}  ({arah} {delta:+.4f})")
```

Plus penjelasan kondisional yang jujur ketika hasilnya tidak membaik.

#### Justifikasi

Perbaikan minimal — hanya membuat output mencerminkan kenyataan. Penjelasan tambahan disertakan karena hasil "tidak membaik" bukan kegagalan yang perlu disembunyikan: dengan subgroup 6–11 orang, `L_fairness` memang sangat berisik dan arahnya bisa berbalik hanya karena 1–2 sampel. Menyatakan itu terus terang **lebih kuat** daripada memaksakan narasi perbaikan.

#### Angka sebelum → sesudah

| | Sebelum | Sesudah |
|---|---|---|
| Output | `0.0132 (turun = lebih adil)` ❌ **salah** | `0.0132 (NAIK +0.0078)` ✅ |
| Penjelasan | selalu positif | kondisional + catatan noise subgroup |

---

### Temuan (b) — Satu fitur bocor masih mendominasi SHAP

#### Apa yang salah

Setelah tiga fitur `IF_*` dibuang (Bug 1), fitur `IF_ID_Stage` **naik menjadi nomor satu** di SHAP dengan nilai 0.046 — sekitar dua kali lipat fitur berikutnya. Padahal `IF_ID_Stage = ID_overall/100 × n_aktor/5`, yaitu versi terskala dari komponen target.

#### Kenapa itu masalah

Meski tiga `IF_*` sudah dibuang, satu yang tersisa tetap **mengaburkan fitur granular** yang justru ingin dibaca — persis masalah yang ingin diselesaikan. Peringkat SHAP jadi sulit ditafsirkan untuk keperluan Bab IV.

#### Apa yang saya ubah

`IF_ID_Stage` dibuang dari fitur model, **dan diganti dengan komponen bersihnya**: `n_actors_encountered` (jumlah aktor yang benar-benar ditemui).

Menariknya, kolom ini **sudah dihitung** di File 04 lalu langsung dibuang:

```python
interact = interact.merge(n_actors, ...)
interact["IF_ID_Stage"] = (base["ID_overall"]/100) * (interact["n_actors_encountered"]/5)
interact = interact.drop(columns=["n_actors_encountered"])   # <-- dibuang, padahal berguna
```

Baris `drop` itu dihapus, jadi kolomnya kini dipertahankan.

#### Justifikasi

`IF_ID_Stage` menggabungkan dua hal: skor ID (yang bocor) dan kelengkapan proses hukum (yang substantif dan **tidak** bocor). Membuang keseluruhannya akan menghilangkan informasi yang berguna; mempertahankannya membawa kebocoran.

Solusinya: **pisahkan** — buang produknya, pertahankan komponen bersihnya. Informasi "sejauh mana proses hukum dijalani" tetap tersedia bagi model tanpa membawa skor ID.

Kolom `IF_ID_Stage` sendiri **tetap dihitung dan disimpan** di `feature_matrix.csv` karena diminta proposal (Tabel 3.28) dan berguna untuk analisis deskriptif; yang dilarang hanya memakainya sebagai prediktor NREI. Ini didokumentasikan langsung di File 04.

#### Angka sebelum → sesudah

| SHAP 3 teratas | Sebelum *(semua bocor)* | Sesudah *(semua granular)* |
|---|---|---|
| #1 | `IF_VfV_ID` — 0.0597 | `T_ID_t3` — 0.0239 |
| #2 | `IF_FCNC_ID` — 0.0574 | `ECV_Investigator_ID` — 0.0239 |
| #3 | `IF_VfV_FCNC` — 0.0358 | `ECV_CorrectionalOfficer_FCNC` — 0.0235 |

**Tidak ada satu pun fitur `IF_*` tersisa di 15 teratas.** Peringkat SHAP kini murni granular — dan bisa dibaca sebagai temuan: *tahap dan aktor mana yang paling menentukan kategori NREI*. Itu pertanyaan yang sah dan berguna untuk Bab IV.

Perhatikan juga nilainya **jauh lebih rata** (0.0239 vs 0.0235, bukan 0.0597 vs 0.0358) — tanda tidak ada lagi satu fitur yang mendominasi karena menyontek.

---

### Keputusan teknis yang saya ambil sendiri

**Masalah:** File 01 **tidak bisa dijalankan sama sekali** di lingkungan ini:

```
TypeError: check_array() got an unexpected keyword argument 'force_all_finite'.
Did you mean 'ensure_all_finite'?
```

**Penyebab:** pustaka `factor_analyzer` 0.5.1 masih memanggil `check_array(force_all_finite=...)`, argumen yang **dihapus** di scikit-learn ≥ 1.8 (diganti `ensure_all_finite`).

**Sudah diverifikasi:** ini **pre-existing** — notebook asli dari backup gagal identik. Bukan akibat perbaikan saya. Dan 0.5.1 adalah **rilis terakhir** yang tersedia, jadi tidak ada versi lebih baru untuk di-upgrade.

**Yang saya lakukan:** menambahkan *shim* kompatibilitas 20 baris di File 01 sel 1, yang menerjemahkan argumen lama ke yang baru. Di scikit-learn versi lama (misalnya Google Colab), blok ini **otomatis tidak melakukan apa-apa** — jadi notebook tetap jalan di kedua lingkungan.

**Kenapa saya ambil keputusan ini tanpa menanyakan lebih dulu:** tanpa shim, File 01 mustahil dijalankan, sehingga Bug 11 tidak bisa diuji sama sekali. Shim-nya **tidak mengubah semantik analisis apa pun** — hanya menerjemahkan nama argumen — dan sepenuhnya reversibel. Silakan minta saya cabut kalau Anda lebih suka menjalankan File 01 hanya di Colab.

---

## Yang sengaja TIDAK diubah

Sama pentingnya dengan yang diubah. Berikut hal-hal yang saya biarkan, beserta alasannya.

| Hal | Nilai | Alasan tidak diubah |
|---|---|---|
| **Flip-rate Monte Carlo** | 6.77% *(di atas ambang 5%)* | **Bukan bug.** Ini temuan sah, dan notebook aslinya sudah menafsirkannya dengan benar: ranking responden stabil (Spearman 0.99), yang rapuh hanya klasifikasi di sekitar ambang 40/60 — wajar untuk sistem berbasis ambang mana pun. |
| **Cronbach's alpha tinggi** | VfV 0.927, FCNC 0.927, ID 0.930 | Artefak cara data dummy dibangkitkan, dan notebook sudah mengakuinya jujur. Di data riil, 0.70–0.85 lebih realistis. |
| **File 00 (Data Cleaning)** | tidak disentuh | Satu-satunya isunya (flag straight-lining tak dipakai) diperbaiki di **File 05** dengan memakai flag-nya, bukan dengan mengubah File 00. |
| **Rumus MAS / ECI / NREI** | tidak diubah | Berasal dari desain proposal. Mengubahnya = mengubah kerangka teori, jauh di luar wewenang perbaikan bug. Yang dilakukan hanya mendokumentasikan konsekuensinya (Bug 15). |
| **Bobot ERM final** | tidak ditetapkan | Proposal mensyaratkan expert panel 2 tahap. Skrip tidak boleh mendahului itu — justru itu inti perbaikan Bug 13. |
| **Kolom `IF_*` di `feature_matrix.csv`** | tetap disimpan | Diminta eksplisit proposal (Tabel 3.28) dan berguna untuk analisis deskriptif. Yang dilarang hanya memakainya sebagai prediktor NREI. |
| **Kebocoran ECV di SET B** | dibiarkan, tapi **diumumkan** | Disengaja. SET B memang dilabeli "konsistensi internal, bukan prediksi", dan auditnya mencetak korelasi 1.0 apa adanya. Menutupi kebocoran jauh lebih berbahaya daripada mengakuinya dengan label yang tepat. |
| **Peringatan kosmetik** | dibiarkan | `tqdm IProgress`, `X does not have valid feature names` — pre-existing, tidak memengaruhi hasil. |

---

## Cara memverifikasi sendiri

### 1. Jalankan dua tes regresi

```bash
cd "C:/Disertasi baru Pak Nanan/Coding"
python tests/test_leakage.py     # Bug 1
python tests/test_bundle.py      # Bug 2, 4, 5
```

**`test_leakage.py`** mencoba semua jalur rekonstruksi pada tiap himpunan fitur, lalu memastikan dua hal sekaligus: (a) SET B lama **terbukti** bocor — ini membuktikan bug-nya nyata, bukan karangan; (b) SET C **tidak** bocor — ini membuktikan perbaikannya bekerja.

**`test_bundle.py`** memeriksa bahwa model yang disimpan cocok dengan juara yang dilaporkan, bahwa `X_train`/`X_test` ada dan tidak beririsan, bahwa tidak ada NaN tersisa, dan bahwa fitur `IF_*` sudah keluar dari daftar.

Keluaran yang diharapkan: **`SEMUA TES LOLOS`** pada keduanya.

### 2. Jalankan ulang seluruh notebook

```bash
cd "C:/Disertasi baru Pak Nanan/Coding/files"
for nb in 01_validity_reliability 03_sensitivity_analysis 04_feature_engineering \
          05_ml_analytics 06_explainability_erm 07_dashboard_export; do
  python -m nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=1500 $nb.ipynb
done
```

Urutannya penting: **04 harus sebelum 05**, dan **05 sebelum 06 sebelum 07**, karena tiap file memakai keluaran file sebelumnya.

> Catatan: pesan `KeyError ... joblib_memmapping_folder` yang mungkin muncul saat menjalankan File 05 adalah **noise pembersihan Windows** dari pemrosesan paralel (`n_jobs=-1`), **bukan kegagalan**. Selama `nbconvert` menulis notebooknya, runnya sukses.

### 3. Bandingkan dengan kondisi sebelum perbaikan

Seluruh kondisi pra-perbaikan tersimpan utuh di `files_backup_pre_fix/` (27 berkas). Metrik "sebelum" dari File 05 terekam di `_before_runs/_BEFORE_05.ipynb`.

---

## Daftar berkas

### Notebook yang diubah

| Berkas | Bug yang diperbaiki |
|---|---|
| `files/01_validity_reliability.ipynb` | 11 + shim kompatibilitas |
| `files/02_index_construction.ipynb` | 15 *(markdown saja)* |
| `files/03_sensitivity_analysis.ipynb` | 6 |
| `files/04_feature_engineering.ipynb` | 12, temuan (b) |
| `files/05_ml_analytics.ipynb` | 1, 2, 3, 4, 7, 8, 9, 10, 12, 14, temuan (a), (b) |
| `files/06_explainability_erm.ipynb` | 2, 5, 13 |
| `files/07_dashboard_export.ipynb` | dijalankan ulang untuk propagasi |

`files/00_data_cleaning.ipynb` **tidak diubah**.

### Berkas baru

| Berkas | Isi |
|---|---|
| `tests/test_leakage.py` | Tes regresi Bug 1 |
| `tests/test_bundle.py` | Tes regresi Bug 2, 4, 5 |
| `files/feature_set_comparison.csv` | Metrik repeated-CV per himpunan fitur + kolom `boleh_disebut` |
| `files/sensitivity_sobol_bounds_lama.csv` | Hasil Sobol versi lama, sebagai pembanding |
| `files_backup_pre_fix/` | Seluruh 27 berkas kondisi pra-perbaikan |
| `_before_runs/_BEFORE_05.ipynb` | Bukti metrik "sebelum" File 05 |

### Lingkungan yang dipakai

```
Python 3.14 (Windows)   numpy 2.4.6      pandas 3.0.3      scipy 1.18.1
scikit-learn 1.9.1      xgboost 3.4.1    shap 0.52.0       factor-analyzer 0.5.1
SALib                   lime             joblib 1.6.0      nbconvert 7.17.1
```

> Notebook aslinya dijalankan di Google Colab (Python 3.12). Karena versi pustakanya berbeda, metrik "sebelum" pada laporan ini diambil dengan **menjalankan ulang notebook asli di lingkungan yang sama** — supaya perbandingannya apel-ke-apel, bukan membandingkan dua lingkungan berbeda.

### Status akhir

- Seluruh 8 notebook dijalankan ulang dari awal sampai akhir: **0 error**
- Dua tes regresi: **keduanya LOLOS**

---

## Ujung terbuka

Satu hal yang **belum** dikerjakan, karena berada di luar daftar bug pada laporan review.

**File 07 masih menyuguhkan angka yang keliru ke dashboard.** Dashboard membaca `model_comparison.csv` — angka dari split tunggal n=40 — padahal File 05 kini eksplisit menyatakan **jangan** menjadikan angka itu headline (Bug 7). Angka yang semestinya tampil sudah tersedia di `feature_set_comparison.csv`, lengkap dengan kolom `boleh_disebut` yang menerangkan himpunan mana yang sah disebut "prediksi".

Akibatnya, kalau dashboard dipakai apa adanya, ia akan menampilkan akurasi ~0.87 sebagai "hasil ML" — angka yang justru sudah kita simpulkan tidak layak dijadikan klaim.

Perbaikannya sekitar 5 baris di File 07. Saya belum mengerjakannya karena tidak ada di laporan review, dan prinsip kerja perbaikan ini adalah tidak mengubah hal di luar daftar. **Beri tahu kalau Anda ingin saya kerjakan.**

---

## Penutup

**Kabar baiknya:** kerangka besar penelitian ini benar sejak awal. Urutan kerjanya — bersihkan data → uji kuesioner → bangun indeks → uji ketahanan → machine learning → penjelasan model → dashboard — sudah sesuai alur baku, kodenya berjalan, dan penulisnya jujur di banyak tempat soal keterbatasan data dummy. Tidak ada satu pun perbaikan di dokumen ini yang menuntut perubahan kerangka.

**Yang perlu diluruskan** ternyata bukan overfitting klasik — itu justru tidak terjadi di sini. Masalah utamanya adalah **kebocoran kunci jawaban** (Bug 1), yang membuat angka akurasi terlihat meyakinkan tanpa membuktikan apa pun, ditambah satu bug model tertukar (Bug 2) dan beberapa kebiasaan teknis yang menyalahi kaidah (Bug 3, 4, 6, 7).

**Tiga hal yang sekarang lebih kuat untuk dipertahankan di sidang:**

1. **Ada model prediksi yang jujur.** SET C (0.423 vs baseline 0.543) menunjukkan pipeline tidak mengarang sinyal pada data acak. Temuan negatif yang sah jauh lebih kuat daripada akurasi tinggi yang tidak bisa dipertanggungjawabkan.

2. **Struktur kuesionernya terbukti kokoh.** Rotasi promax menunjukkan cross-loading yang dulu dikhawatirkan memang artefak (0.22–0.30 → 0.00–0.06), parallel analysis mengonfirmasi 3 faktor, dan EFA pembanding pada 129 responden independen menjawab keberatan pseudo-replikasi **dengan data**, bukan dengan catatan kaki.

3. **Temuan sensitivity-nya bertahan, dengan angka yang benar.** Bobot NREI memang paling berpengaruh — tapi 7.94×, bukan 34×.

Sisi lain yang tak kalah berharga: notebook kini **memeriksa dirinya sendiri**. Audit kebocoran berjalan otomatis setiap run, `assert` menggagalkan notebook kalau SET C tercemar, kestabilan cluster dinilai eksplisit dan direkam ke CSV, dan setiap klaim arah perubahan dibaca dari data alih-alih ditulis permanen. Kalau bug-bug ini terulang di data riil nanti, mereka akan **berteriak**, bukan diam-diam lolos.
