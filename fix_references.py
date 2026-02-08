import os
import re

ROOT_DIR = r"d:\ALL IN AI\metauniverse-blog\docs\knowledge\cs336"

def process_lecture(lecture_dir):
    images_dir = os.path.join(lecture_dir, "images")
    if not os.path.exists(images_dir):
        return

    # Map standardized names to potential old names
    mapping = {}
    for filename in os.listdir(images_dir):
        if re.match(r'l\d+-', filename):
            suffix = re.sub(r'^l\d+-', '', filename)
            # 1. The pure suffix
            mapping[suffix] = filename
            # 2. The "lectureN-" prefixed version
            lecture_num = lecture_dir.split("Lecture")[-1]
            mapping[f"lecture{lecture_num}-{suffix}"] = filename
            # 3. Special case for L11 which had numbered slides
            # No additional mapping needed as suffix already handles it for L11

    for root, dirs, files in os.walk(lecture_dir):
        if "images" in dirs:
            dirs.remove("images")
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                print(f"Processing {filepath}...")
                
                content = None
                for enc in ['gbk', 'utf-8-sig', 'utf-8']:
                    try:
                        with open(filepath, 'r', encoding=enc) as f:
                            content = f.read()
                        break
                    except:
                        continue
                
                if content is None:
                    print(f"Failed to read {filepath}")
                    continue

                original_content = content
                
                # Sort mapping keys by length descending to avoid partial replacements
                # (e.g., 'data.png' vs 'long-data.png')
                sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
                
                for old_name in sorted_keys:
                    new_name = mapping[old_name]
                    # Replace only if it's in a path like images/old_name
                    # or in a markdown image tag.
                    # We look for 'images/' + old_name
                    content = content.replace(f"images/{old_name}", f"images/{new_name}")
                    content = content.replace(f"./images/{old_name}", f"./images/{new_name}")

                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Updated references in {filepath}")

for i in range(1, 18):
    lec_dir = os.path.join(ROOT_DIR, f"Lecture{i}")
    if os.path.exists(lec_dir):
        process_lecture(lec_dir)
