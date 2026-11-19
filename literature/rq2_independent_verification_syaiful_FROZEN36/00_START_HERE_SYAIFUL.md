# Verifikasi independen RQ2 oleh Syaiful Anam

## Tujuan

Lakukan coding independen terhadap **frozen sample 36 paper** yang digunakan dalam audit RQ2: 24 paper
dari strata top-cited dan 12 paper dari strata random. Coding Hilmi sengaja tidak disertakan dalam paket
ini.

## Urutan kerja

1. Baca `01_CODEBOOK.md` sampai selesai.
2. Lihat `02_EXAMPLE_COMPLETED_ROW.csv` hanya untuk memahami format. Baris tersebut fiktif dan bukan
   bagian dari sample.
3. Kerjakan hanya di `03_CODING_SHEET_SYAIFUL.csv`.
4. Cocokkan setiap `paper_id` dengan dokumen dan status akses pada `04_PAPERS_AND_ACCESS.md`.
5. Isi seluruh tujuh keputusan, evidence, `coder`, catatan, dan `completion_status`.
6. Jalankan pemeriksaan pada `05_RETURN_CHECKLIST.md` sebelum mengembalikan file.

## Aturan independensi

Sebelum seluruh coding selesai dan file dikembalikan, jangan membuka atau meminta isi dari:

- `literature/syssearch/rq2_coding.csv`;
- `literature/syssearch/SOURCE_CODING_SYAIFUL.csv`;
- file yang namanya diawali `literature_audit_hilmi`;
- tabel atau narasi hasil audit RQ2 di manuscript;
- jawaban atau klasifikasi yang telah dibuat Hilmi.

Jangan mendiskusikan keputusan per paper dengan Hilmi sebelum file dikunci. Jika ada ketidakjelasan,
tulis di `general_notes`; ketidakjelasan tersebut dibahas saat reconciliation setelah coding independen
selesai.

## Sumber full text

- Sebanyak 18 full text tersedia sebagai PDF di `fulltext_pdf/`, dan satu full text open-access tersedia
  sebagai teks/XML di `fulltext_txt/`.
- Sebanyak 17 full text lainnya harus dibuka melalui DOI; sebagian open access dan sebagian memerlukan
  institutional access. Daftarnya terdapat di `06_FULLTEXT_ACCESS_REQUIRED.md`.
- Jangan melakukan coding dari abstract, preview, metadata, atau coding Hilmi.
- Untuk paper yang belum dibundel, gunakan versi artikel penerbit dan, jika diunduh, simpan menggunakan
  nama yang sudah ditentukan, misalnya `fulltext_pdf/10.pdf`.
- Jika full text tidak bisa diperoleh, jangan menebak. Isi `completion_status=blocked` dan jelaskan pada
  `general_notes`.

## Cara mengisi coding sheet

1. Isi `coder=SA`.
2. Gunakan persis kategori yang diperbolehkan dalam codebook; jangan menambahkan uraian ke dalam sel
   kategori.
3. Masukkan uraian, kutipan, dan lokasi sumber ke kolom evidence yang sesuai.
4. Untuk keputusan positif atau partial, sertakan kutipan singkat serta halaman, section, table, atau
   appendix.
5. Untuk keputusan `no`, `not stated`, atau bentuk ketiadaan lain, tulis bagian dan tabel yang sudah
   diperiksa.
6. Isi `completion_status=complete` hanya setelah seluruh tujuh keputusan pada baris tersebut selesai.
7. Jangan menghapus, menambah, mengurutkan ulang, atau mengganti nama kolom/baris.

## File yang dikembalikan

Kembalikan:

1. `03_CODING_SHEET_SYAIFUL.csv` yang telah diisi; dan
2. daftar singkat paper yang berstatus `blocked`, jika ada.

Jika Syaiful mengunduh salah satu full text institutional-access, simpan secara privat. PDF dapat
dikembalikan kepada Hilmi melalui kanal privat jika kebijakan akses institusi mengizinkan, tetapi jangan
diunggah ke repositori publik.

Setelah file dikembalikan dan dikunci, kedua coding sheet akan dibandingkan menggunakan raw agreement dan
Cohen's kappa per pertanyaan. Perbedaan dibahas setelah itu dan dicatat tanpa menimpa jawaban asli.
