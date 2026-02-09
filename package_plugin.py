
import zipfile
import os

source_root = r"c:\Users\nuri9\AI-Vectorizer-for-Archaeology"
plugin_dir = os.path.join(source_root, "ai_vectorizer")
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
zip_path = os.path.join(desktop, "ArchaeoTrace_v0.1.1_Essential.zip")

print(f"Creating ZIP at {zip_path}")
print(f"Source: {plugin_dir}")

try:
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add plugin files recursively
        count = 0
        total_size = 0
        for root, dirs, files in os.walk(plugin_dir):
            # Exclude directories
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            if 'models' in dirs:
                # Exclude entire models folder (user wants < 25MB, model is 40MB)
                dirs.remove('models')
                
            for file in files:
                if file.endswith('.pyc'): continue
                
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, source_root) # ai_vectorizer/...
                
                zipf.write(abs_path, rel_path)
                count += 1
                total_size += os.path.getsize(abs_path)
        
        # Add README and LICENSE from root to plugin folder in zip
        # This ensures users see them even if they unzip just the folder
        readme_src = os.path.join(source_root, "README.md")
        license_src = os.path.join(source_root, "LICENSE")
        
        if os.path.exists(readme_src):
            zipf.write(readme_src, "ai_vectorizer/README.md")
            count += 1
        
        if os.path.exists(license_src):
            zipf.write(license_src, "ai_vectorizer/LICENSE")
            count += 1
            
    final_size = os.path.getsize(zip_path) / (1024*1024)
    print(f"SUCCESS: Created ZIP with {count} files.")
    print(f"Final Size: {final_size:.2f} MB")

except Exception as e:
    print(f"ERROR: {e}")
