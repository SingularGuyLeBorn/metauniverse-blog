import os
import sys

base_path = r"d:\ALL IN AI\metauniverse-blog\docs\knowledge\llm-galaxy"

src_name = "0-2. 核心原理与架构 (Core Principles & Architecture)"
dst_name = "2. 核心原理与架构 (Core Principles & Architecture)"

src = os.path.join(base_path, src_name)
dst = os.path.join(base_path, dst_name)

print(f"Attempting to rename:\n'{src}'\n->\n'{dst}'")

if not os.path.exists(src):
    print("ERROR: Source path does not exist!")
    # List dir to see what IS there
    print("Available contents:")
    for x in os.listdir(base_path):
        print(f" - {x}")
else:
    try:
        os.rename(src, dst)
        print("SUCCESS: Renamed successfully.")
    except Exception as e:
        print(f"FAILED: {e}")
        # explicit error type
        print(type(e))
