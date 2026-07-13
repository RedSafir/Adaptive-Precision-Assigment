# Reduksi Biaya Komputasi menggunakan Adaptive Precision Layering untuk Image Classification

Repositori ini menyimpan codebase untuk penelitian skripsi yang berfokus pada optimasi memori GPU (VRAM) dan efisiensi komputasi pada pelatihan model pembelajaran mendalam (*deep learning*) untuk klasifikasi gambar. Metode yang diimplementasikan adalah **Adaptive Precision Assignment (APA)** yang dinamis dan terakselerasi secara perangkat keras (*hardware-accelerated*).

---

## 🚀 Evolusi Arsitektur: Native Hardware-Accelerated vs. Simulated Precision

Proyek ini merupakan evolusi dan peningkatan signifikan dari repositori referensi asli [wonyeol/mixed-prec-train](https://github.com/wonyeol/mixed-prec-train). Perbedaan mendasar terletak pada bagaimana presisi rendah (*low precision*) dieksekusi:

*   **Arsitektur Referensi (Simulasi / Bit-Truncation):**
    Menggunakan library `qtorch3` untuk menyimulasikan pemotongan bit (*bit truncation*) di dalam wadah tensor FP32. Pendekatan simulasi ini **tidak menghasilkan penghematan memori VRAM riil** ataupun akselerasi kecepatan eksekusi pada hardware GPU karena secara fisik operasi matmul tetap berjalan dalam presisi FP32 penuh.
*   **Arsitektur Proyek Ini (Native Hardware-Accelerated):**
    Mengimplementasikan **Native Hardware-Accelerated Precision** dengan memanfaatkan tipe data bawaan PyTorch 2.1+ (`torch.float8_e4m3fn` untuk *forward pass*, `torch.float8_e5m2` untuk *backward pass*, `torch.float16`, dan `TF32`). Pendekatan ini memberikan **reduksi alokasi memori VRAM riil** dan memanfaatkan unit komputasi khusus (Tensor Cores) pada arsitektur GPU modern seperti NVIDIA Ada Lovelace (Compute Capability 8.9) dan Hopper (Compute Capability 9.0+).

---

## 🧠 3 Pilar Utama Mekanisme APA (Adaptive Precision Assignment)

Arsitektur APA di dalam repositori ini beroperasi berdasarkan tiga mekanisme inti yang menjaga keseimbangan antara stabilitas konvergensi model dan efisiensi memori GPU:

### 1. Block-Based Tensor Grouping
Layer komputasi (konvolusi) tidak dievaluasi atau diubah presisinya satu per satu secara independen. Sebaliknya, manajer APA mengelompokkannya ke dalam satu **Group ID** yang sama berdasarkan blok arsitektur komputasinya. 
*   **Logika Pengelompokan:** Layer linear (`NativeLinear`) selalu menginisialisasi grup baru, sedangkan layer konvolusi (`NativeConv2d`) digabungkan ke dalam Group ID yang sama selama mereka memiliki jumlah *input channels* (`in_channels`) yang identik. Layer pembantu seperti BatchNorm, ReLU, dan MaxPool mewarisi Group ID dari layer komputasi sebelumnya.

### 2. Comprehensive Demotion (Activation + Parameter)
Saat sebuah grup terpilih oleh scheduler APA untuk didegradasi presisinya demi menghemat memori (*demoted*), sistem secara simultan menurunkan presisi dari kedua sisi topologi graph:
*   **Aktivasi (Ttype.Y & Ttype.GY):** Output forward dan gradien backward dari layer didegradasi ke FP8.
*   **Parameter/Bobot (Ttype.P & Ttype.GP):** Bobot tensor komputasi beserta gradien bobotnya didegradasi ke presisi FP8 menggunakan *delayed scaling manager* dengan amax history circular buffer untuk meminimalkan fluktuasi magnitudo.

### 3. Zero-Cost Overflow Promotion
Untuk mencegah terjadinya ketidakstabilan numerik (*numerical instability*) atau gradien meledak (*exploded gradients*) yang berujung pada nilai `NaN`, sistem menginjeksikan pemantauan overflow berbiaya nol (*zero-cost overflow tracking*) secara *on-the-fly* pada setiap operasi komputasi FP8:
*   **Metrik Rasio:** Sistem menghitung rasio overflow riil pada aktivasi dan bobot parameter terhadap total elemen tensor:
    $$\text{Rasio Overflow} = \frac{\sum (|T| > V_{\text{max}})}{N_{\text{elemen}}}$$
    Di mana $V_{\text{max}}$ (Nilai Maksimal Representasi) untuk format FP8 E4M3 bernilai **448.0**.
*   **Promosi Dinamis:** Jika rasio overflow pada salah satu representasi tensor melewati ambang batas (`ovr_thrs`), grup layer tersebut akan **dipromosikan kembali ke presisi tinggi** (TF32 atau FP16) secara permanen untuk iterasi-iterasi selanjutnya guna mengamankan konvergensi model.

---

## 📁 Struktur Repositori

```
├── ext3/
│   ├── core/                  # Logika inti graph, Dtype, Pasn, dan EModlObjMgr
│   │   ├── emodlobj/
│   │   │   └── emodlobjmgr.py # Manajer objek, estimasi VRAM, dan logika promosi
│   │   └── include/
│   │       ├── native_precision.py # Context manager, casting helper, dan FP8 scaling manager
│   │       └── ...
│   └── nn/
│       ├── nn_base.py         # Layer dasar EModl
│       ├── nn_native.py       # Wrapper layer native precision (NativeConv2d, NativeLinear, dll.)
│       └── __init__.py
├── rewrite_notebooks.py       # Script otomatisasi penulisan ulang notebook
├── vgg16_cifar10_tf32_fp8.ipynb # Notebook pelatihan VGG16 dengan baseline TF32 vs FP8
├── vgg16_cifar10_fp16_fp8.ipynb # Notebook pelatihan VGG16 dengan baseline FP16 vs FP8
└── README.md
```

---

## 🛠️ Panduan Menjalankan Riset (How to Run)

Seluruh alur pelatihan, monitoring alokasi VRAM secara riil, pencatatan statistik overflow, dan plot evaluasi dipusatkan di dalam Jupyter Notebooks.

### Prasyarat System & Software:
1.  **GPU Modern:** NVIDIA RTX 40-Series (Ada Lovelace) atau H100 (Hopper) ke atas untuk dukungan penuh hardware-accelerated FP8 Tensor Cores.
2.  **Software Stack:**
    *   CUDA Toolkit >= 12.0
    *   Python >= 3.10
    *   PyTorch >= 2.1 (direkomendasikan versi terbaru untuk performa `_scaled_mm` optimal)
    *   Jupyter Notebook / JupyterLab
    *   Matplotlib, NumPy, dan Torchvision

### Langkah Eksekusi:
1.  Jalankan script regenerasi notebook untuk memastikan template notebook sinkron dengan pembaruan backend layer di `ext3`:
    ```bash
    python rewrite_notebooks.py
    ```
2.  Buka Jupyter Notebook pilihan Anda sesuai baseline yang ingin diuji:
    *   Untuk baseline TF32 vs FP8: Buka dan jalankan `vgg16_cifar10_tf32_fp8.ipynb`
    *   Untuk baseline FP16 vs FP8 (Autocast + GradScaler): Buka dan jalankan `vgg16_cifar10_fp16_fp8.ipynb`
3.  Jalankan cell secara berurutan mulai dari inisialisasi lingkungan (*environment check*), penentuan konfigurasi scheduler APA, pembuatan model `VGG16Native`, hingga proses pelatihan utama.
