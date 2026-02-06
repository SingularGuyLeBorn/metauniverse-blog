import os
import re

base_path = r"d:\ALL IN AI\metauniverse-blog\docs\knowledge\llm-galaxy"

def fix_folders():
    if not os.path.exists(base_path):
        print("Base path not found.")
        return

    items = os.listdir(base_path)
    
    # We want to match "0-1. xxx" and rename to "01. xxx" or just "1. xxx"
    # Actually, user's list was "1. 导论", "2. ...", "10. ..."
    # So we should rename "0-1." -> "1.", "0-10." -> "10."
    
    for item in items:
        # Match pattern: 0-X. Title
        match = re.match(r'^0-(\d+)\.(.*)', item)
        if match:
            num = match.group(1)
            rest = match.group(2)
            
            # Construct new name: "1. 导论..."
            # Pad simple numbers to 01 for better file system sorting visually? 
            # User's request showed "1.", "1.1".
            # The sidebar natural sort handles "1." vs "10." correctly now.
            # But let's use "01." etc for cleaner File Explorer view if we want.
            # However, matching the user's exact "1. 导论" is safest.
            
            new_name = f"{num}.{rest}"
            
            src = os.path.join(base_path, item)
            dst = os.path.join(base_path, new_name)
            
            print(f"Renaming: {item} -> {new_name}")
            try:
                os.rename(src, dst)
            except Exception as e:
                print(f"Error renaming {item}: {e}")

if __name__ == "__main__":
    fix_folders()
