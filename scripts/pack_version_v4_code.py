import os
import zipfile

def pack_code():
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    zip_path = os.path.join(base_dir, "IPS_ThacSi_V4_Code.zip")
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    include_dirs = ["src", "scripts", "configs", "docs", "tests", "examples"]
    include_files = ["pyproject.toml", "requirements.txt", "README.md", "AGENTS.md", ".gitignore"]
    
    exclude_extensions = {".pyc", ".pyo", ".pyd"}
    exclude_dirs = {"__pycache__", ".venv", ".venv39", ".git", "runs", ".archify", ".vscode", ".idea"}
    
    print(f"Creating ZIP archive: {zip_path}")
    count = 0
    total_uncompressed_bytes = 0
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # Add directories
        for d in include_dirs:
            full_d = os.path.join(base_dir, d)
            if not os.path.exists(full_d):
                continue
            for root, dirs, files in os.walk(full_d):
                # Filter out excluded directories
                dirs[:] = [sub_d for sub_d in dirs if sub_d not in exclude_dirs]
                
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in exclude_extensions:
                        continue
                    file_path = os.path.join(root, f)
                    rel_path = os.path.relpath(file_path, base_dir).replace("\\", "/")
                    zipf.write(file_path, arcname=rel_path)
                    count += 1
                    total_uncompressed_bytes += os.path.getsize(file_path)
                    
        # Add root files
        for f in include_files:
            file_path = os.path.join(base_dir, f)
            if os.path.exists(file_path):
                rel_path = f
                zipf.write(file_path, arcname=rel_path)
                count += 1
                total_uncompressed_bytes += os.path.getsize(file_path)
                
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)
    print(f"Successfully packed {count} files.")
    print(f"Uncompressed size: {raw_size_mb:.2f} MB")
    print(f"Compressed ZIP size: {zip_size_mb:.2f} MB")
    print(f"Saved at: {zip_path}")

if __name__ == "__main__":
    pack_code()

