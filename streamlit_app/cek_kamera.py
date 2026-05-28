import cv2

print("Memeriksa ketersediaan kamera di laptop Anda...")
print("Mohon tunggu sebentar...\n")

for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"[✓] Kamera Index {i} : BISA DIAKSES")
        # Jika bisa, tes membaca 1 frame
        ret, frame = cap.read()
        if ret:
            print(f"    -> Berhasil membaca gambar beresolusi: {frame.shape}")
        else:
            print(f"    -> Namun GAGAL mengambil gambar (frame kosong).")
        cap.release()
    else:
        print(f"[X] Kamera Index {i} : TIDAK DITEMUKAN / DIBLOKIR")

print("\nSelesai pengecekan.")
