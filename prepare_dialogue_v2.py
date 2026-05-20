"""
合并 Cornell + DailyDialog → 独立 Q&A 对
"""
import os, re, random, urllib.request

def load_cornell():
    """Cornell Movie-Dialogs"""
    raw_dir = os.path.join("raw_data", "cornell_movie_dialogs", "cornell movie-dialogs corpus")
    with open(os.path.join(raw_dir, "movie_lines.txt"), "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    line_to_text = {}
    for m in re.findall(r"^(L\d+) \+\+\+\$\+\+\+ .+? \+\+\+\$\+\+\+ .+? \+\+\+\$\+\+\+ .+? \+\+\+\$\+\+\+ (.+)$", raw, re.M):
        line_to_text[m[0]] = m[1].strip()

    with open(os.path.join(raw_dir, "movie_conversations.txt"), "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    pairs = set()
    for ids_str in re.findall(r"\['(.+?)'\]", raw):
        ids = ids_str.split("', '")
        for i in range(len(ids) - 1):
            q = line_to_text.get(ids[i], "")
            a = line_to_text.get(ids[i + 1], "")
            if len(q) > 2 and len(a) > 2 and q != a:
                pairs.add(f"Q: {q}\nA: {a}\n")
    return pairs

def load_dailydialog():
    """DailyDialog — 从原始 tsv 直接下载解析"""
    url = "http://yanran.li/files/ijcnlp_dailydialog.zip"
    zip_path = os.path.join("raw_data", "dailydialog.zip")
    out_dir = os.path.join("raw_data", "dailydialog")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print("下载 DailyDialog...")
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        os.remove(zip_path)
        print("DailyDialog 下载完成")

    pairs = set()
    for split in ["train", "validation", "test"]:
        dialog_file = os.path.join(out_dir, "ijcnlp_dailydialog", split, f"dialogues_{split}.txt")
        if not os.path.exists(dialog_file):
            continue
        with open(dialog_file, "r", encoding="utf-8") as f:
            for line in f:
                turns = [t.strip() for t in line.split("__eou__") if t.strip()]
                for i in range(len(turns) - 1):
                    q, a = turns[i], turns[i + 1]
                    if len(q) > 2 and len(a) > 2 and q != a and "http" not in q and "http" not in a:
                        pairs.add(f"Q: {q}\nA: {a}\n")
    return pairs

# 执行
cornell = load_cornell()
print(f"Cornell: {len(cornell)} 对")

try:
    daily = load_dailydialog()
    print(f"DailyDialog: {len(daily)} 对")
except Exception as e:
    print(f"DailyDialog 失败: {e}")
    daily = set()

all_pairs = list(cornell | daily)
random.shuffle(all_pairs)

output_path = os.path.join("raw_data", "dialogue_text.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_pairs))

print(f"合计: {len(all_pairs)} 对")
print(f"文件: {output_path}  ({len(all_pairs)} 行)")

print("\n--- 样本 ---")
for s in random.sample(all_pairs, 3):
    print(s)
