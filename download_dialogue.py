import requests
import zipfile
import os

url = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
raw_dir = os.path.join("raw_data", "cornell_movie_dialogs")
os.makedirs(raw_dir, exist_ok=True)
zip_path = os.path.join(raw_dir, "cornell_movie_dialogs_corpus.zip")

print("下载中（大文件，耐心等）...")
response = requests.get(url, stream=True, timeout=120)
with open(zip_path, 'wb') as f:
  for chunk in response.iter_content(chunk_size=8192):
      f.write(chunk)

print(f"下载完成: {os.path.getsize(zip_path)} bytes")
print("解压中...")

with zipfile.ZipFile(zip_path, 'r') as zf:
  zf.extractall(raw_dir)
os.remove(zip_path)

print(f"解压完成 → {raw_dir}/")
extracted = os.path.join(raw_dir, "cornell movie-dialogs corpus")
for f in sorted(os.listdir(extracted)):
    print(f"  {f}")