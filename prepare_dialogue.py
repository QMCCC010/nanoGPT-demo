"""
把 Cornell Movie-Dialogs Corpus 转成 Q&A 对话格式
输出: raw_data/dialogue_text.txt
"""
import os
import re

raw_dir = os.path.join("raw_data", "cornell_movie_dialogs", "cornell movie-dialogs corpus")
lines_file = os.path.join(raw_dir, "movie_lines.txt")
conv_file = os.path.join(raw_dir, "movie_conversations.txt")

# ==============================================
# 1. 加载 lineID → text
# ==============================================
line_to_text = {}
line_to_char = {}
with open(lines_file, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 格式: L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
        parts = line.split(" +++$+++ ")
        if len(parts) >= 5:
            line_id = parts[0]
            char_name = parts[3].strip()
            text = parts[4].strip()
            line_to_text[line_id] = text
            line_to_char[line_id] = char_name

print(f"加载了 {len(line_to_text)} 行台词")

# ==============================================
# 2. 加载对话，转成 Q&A（按最终说话者交替标记）
# ==============================================
dialogues = []
with open(conv_file, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" +++$+++ ")
        if len(parts) >= 4:
            ids_str = parts[3]
            ids = re.findall(r"L\d+", ids_str)
            # 收集这段对话中的所有台词
            turns = []
            for lid in ids:
                if lid in line_to_text:
                    text = line_to_text[lid]
                    if len(text) > 2:
                        turns.append(text)
            # 交替标记: 第一句 Q，第二句 A，第三句 Q，第四句 A...
            if len(turns) >= 2:
                lines = []
                for i, text in enumerate(turns):
                    tag = "Q" if i % 2 == 0 else "A"
                    lines.append(f"{tag}: {text}")
                dialogues.append("\n".join(lines))

print(f"生成了 {len(dialogues)} 段对话")

# ==============================================
# 3. 写入文件
# ==============================================
output_path = os.path.join("raw_data", "dialogue_text.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(dialogues))

# 统计
with open(output_path, "r", encoding="utf-8") as f:
    content = f.read()
print(f"输出文件: {output_path}")
print(f"文件大小: {len(content)} 字符")
print(f"唯一字符: {len(set(content))}")

# 打印第一段对话
print("\n--- 样本 ---")
print(dialogues[0] if dialogues else "None")
