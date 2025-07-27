import os
import subprocess

def download_folder(gdrive_url, target_dir):
    print(f"📥 Downloading from: {gdrive_url}")
    command = [
        "gdown",
        "--folder",
        "--id", extract_folder_id(gdrive_url),
        "-O", target_dir
    ]
    subprocess.run(command)

def extract_folder_id(url):
    # Extract the folder ID from Google Drive folder link
    if "folders/" in url:
        return url.split("folders/")[1].split("?")[0]
    raise ValueError("Invalid Google Drive folder URL.")

def main():
    folders = {
        "https://drive.google.com/drive/folders/17HxRkuszSbYGkHEx17d-lh7YSzZaLJe8?usp=drive_link": "detector/bert_model",
        "https://drive.google.com/drive/folders/1oIO69AQIltKTTI4phKKqH2GzIHefxxMI?usp=drive_link": "detector/fine_tuned_bert"
    }

    for url, out_dir in folders.items():
        os.makedirs(out_dir, exist_ok=True)
        download_folder(url, out_dir)

    print("✅ All models downloaded into detector/")

if __name__ == "__main__":
    main()
