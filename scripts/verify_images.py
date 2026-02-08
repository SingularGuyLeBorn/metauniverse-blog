import os
import re
import urllib.parse

def verify_images(root_dir):
    print(f"Scanning {root_dir}...")
    image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    
    broken_links = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.md'):
                md_path = os.path.join(dirpath, filename)
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all image links
                matches = image_pattern.findall(content)
                for link in matches:
                    # Ignore external links
                    if link.startswith('http') or link.startswith('https'):
                        continue
                    
                    # Clean link (remove query params, fragments)
                    clean_link = urllib.parse.unquote(link.split(' ')[0])
                    
                    # Resolve absolute path
                    if clean_link.startswith('/'):
                        # Assuming / relative to docs root, but here we might need to be careful
                        # Let's assume relative paths mostly
                        pass 
                    
                    # Resolve relative path
                    target_path = os.path.normpath(os.path.join(dirpath, clean_link))
                    
                    if not os.path.exists(target_path):
                        broken_links.append({
                            'file': md_path,
                            'link': link,
                            'target': target_path
                        })

    if broken_links:
        print(f"Found {len(broken_links)} broken image links:")
        for error in broken_links:
            print(f"[BROKEN] In {error['file']}:")
            print(f"  Link: {error['link']}")
            print(f"  Target: {error['target']}")
            print("-" * 20)
    else:
        print("No broken image links found.")

if __name__ == "__main__":
    verify_images(r"d:\ALL IN AI\metauniverse-blog\docs\knowledge\cs336")
