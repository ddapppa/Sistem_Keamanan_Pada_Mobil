🚗 Real-time Driver Drowsiness Detection System

## 📖 Deskripsi Singkat
Proyek ini adalah implementasi sistem cerdas untuk mendeteksi kantuk pada pengemudi secara *real-time* menggunakan gabungan teknologi **Computer Vision** dan **Deep Learning**. 

Sistem ini dirancang berasitektur **Hybrid Spatial-Temporal**. Wajah pengemudi akan diolah menggunakan fitur pemindai deteksi seperti **Swin Transformer** (terdapat juga varian CNN, MobileNet, VGG) sebagai *Backbone* untuk ekstraksi fitur (spatial). Kemudian, fitur-fitur dari beberapa frame (30 frame berurutan) digabungkan dan diproses secara dinamis menggunakan model **LSTM/BiLSTM** (temporal) untuk mengenali pola waktu, sehingga sistem tidak hanya mendeteksi kantuk dari 1 foto, namun dari serangkaian gerakan aktivitas pengukur kantuk (seperti mata terpejam lama, menguap, dsb).

## 🏢 Alur Proses (Pipeline Sistem)
1. **Analisis Dataset (EDA):** Eksplorasi dataset menggunakan NTHU Drowsy Driver Detection (NTHUD) dan MRL Eye Dataset untuk melatih model membedakan status *drowsy* (mengantuk) dan *not drowsy* (sadar).
2. **Pre-Processing & Mediapipe PIPELINE:** Video dipecah menjadi *frames*, wajah dan mata di-crop (dipusatkan) dengan bantuan alat seperti Google Mediapipe.
3. **Ekstraksi Fitur Statis (Spatial):** Potongan frame dikumpulkan untuk diekstraksi ke dalam fitur array/tensor melalui beragam model arsitektur (*Feature Extractors*).
4. **Penyusunan Sekuensial (Build Sequences):** Pembuatan gabungan sekuens sepanjang **30 frames** per sampel.
5. **Pelatihan Model Temporal (LSTM/BiLSTM):** Sekuens fitur 30 frame dilatih mencari pola waktu terjadinya kantuk melalui algoritma Recurrent (LSTM). Dilakukan juga Hyperparameter Tuning yang cukup intensif guna menemukan kombinasi model yang optimal.
6. **Optimasi Evaluasi (Sistem OpenVINO):** Eksperimen konversi jaringan model menggunakan OpenVINO agar kinerja inferensi lebih ringan dan cepat secara komputasi.
7. **Penyebaran / UI (Streamlit):** Web UI lokal di mana sistem dihubungkan ke Webcam untuk pendeteksian secara real-time berbasis Streamlit disertai pemicu audio (Alarm).

## 📂 Struktur Direktori & Modul
Sistem ini dibangun dengan pendekatan modular menggunakan Jupyter Notebook untuk eksperimen bertahap dan file Python murni untuk sistem operasional. Berikut adalah rincian fungsional file utamanya:
* **`01`, `02_A`, `05`: Analisis Eksploratif (EDA)** - Modul persiapan & penelusuran data pada *MRL Dataset* dan *NTHUD Dataset*.
* **`02_B`, `05` s/d `06`: Pipeline Mediapie & Preprocessing** - Skrip untuk penyiapan konversi aset video dataset dan pelacakan titik krusial wajah secara akurat.
* **`03` : pembuatan model SWIN dan experiment** - Nantinya akan di pakai dalam tahapan LSTM untuk pendeteksi mata.
* **`07_A` & `07_B`**: **Module Ekstraktor Backcbone** - Penyesuaian arsitektur (Swin Transformer / Model CNN lain) untuk mengurai frame piksel menjadi *Frame Features*.
* **08_Build_Sequences_30Frames_NTHU.ipynb**: Konversi fitur terpisah tadi menjadi matriks *time-series* untuk dibaca spesifik LSTM.
* **`09_A` & `09_B` : Tahap Training & Tuning** - Pelatihan arsitektur LSTM/BiLSTM dan percobaan parameter berlapis (Model tuning dan optimasi parameter keselamatan).
* **10_Evaluasi_performa_model.ipynb**: Evaluasi statistik yang meninjau Metrik (Confusion matrix, F1-Score, dan Safety score) dari model terlatih Anda.
* **11_OpenVino.ipynb**: Pendekatan optimasi dan akselerasi format inference engine menggunakan OpenVINO.
* **app.py**: Aplikasi GUI utama berbasis **Streamlit** (Dilengkapi desain *Figma match*, pengaturan bahasa, demo real-time dengan akses webcam, panel metrik, dan *Alarm Manager*).
* **inference.py / inference_jaring.py**: Modul yang bekerja di belakang layar antarmuka pengguna, mengeksekusi model terhadap video secara langsung di luar environment training.

## ✨ Fitur & Fungsi Utama Aplikasi
- [x] **Real-time Monitoring:** Penjadwalan inferensi sangat cepat yang mendukung penggunaan Webcam lokal.
- [x] **Temporal Logic (Jendela 30 Frame):** Mengurangi *False Positive/Alarm palsu* saat Anda sekedar berkedip biasa. Sistem butuh urutan untuk menyimpulkan Anda sungguh mengantuk.
- [x] **Smart Alarm Manager:** Notifikasi bunyi otomatis jika deteksi state *drowsy* tertahan lebih dari *threshold/batas waktu* kritis.
- [x] **Multi-Model Extractor:** Dapat memilih pendekatan menggunakan Swin Transformer (State-of-the-Art), MobileNet (Ringan), atau jaringan konvensional secara modular.
- [x] **Dashboard Interaktif Berbasis Streamlit:** Halaman UI mencakup Beranda, Tentang, Halaman Demo (Visualisasi video), Fitur, serta Kontak dengan tampilan yang bersih, modern dan responsif.
