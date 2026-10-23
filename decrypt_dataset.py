"""
Dataset Decryption Utility for OVAscan AI
Decrypts and restores the 'real time data/' clinical ultrasound dataset using AES-256-GCM.
"""
import os
import sys
import getpass
import zipfile
import io
import argparse

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
except ImportError:
    print("Error: 'cryptography' library is required. Install via: pip install cryptography")
    sys.exit(1)


def decrypt_dataset(enc_file="real_time_data.enc", output_dir="real time data", password=None):
    if not os.path.exists(enc_file):
        print(f"Error: Encrypted file '{enc_file}' not found.")
        sys.exit(1)

    if not password:
        password = getpass.getpass("Enter decryption key / password for OVAscan dataset: ").strip()

    if not password:
        print("Error: Password cannot be empty.")
        sys.exit(1)

    print(f"Reading encrypted archive '{enc_file}'...")
    with open(enc_file, "rb") as f:
        salt = f.read(16)
        nonce = f.read(12)
        ciphertext = f.read()

    print("Deriving cryptographic key (PBKDF2-HMAC-SHA256, 100,000 iterations)...")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000
    )
    key = kdf.derive(password.encode("utf-8"))
    aesgcm = AESGCM(key)

    print("Decrypting with AES-256-GCM...")
    try:
        decrypted_zip_bytes = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception:
        print("\n[FAILED] Decryption failed! Incorrect key/password or corrupted file.")
        sys.exit(1)

    print(f"Extracting dataset to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(decrypted_zip_bytes), "r") as z:
        z.extractall(output_dir)
        count = len(z.namelist())

    print(f"\n[SUCCESS] Successfully decrypted and restored {count} files to '{output_dir}/'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decrypt OVAscan AI clinical ultrasound dataset.")
    parser.add_argument("--key", "-k", type=str, default=None, help="Decryption key / password")
    parser.add_argument("--file", "-f", type=str, default="real_time_data.enc", help="Path to encrypted file")
    parser.add_argument("--out", "-o", type=str, default="real time data", help="Output directory")
    args = parser.parse_args()

    decrypt_dataset(enc_file=args.file, output_dir=args.out, password=args.key)
