import os

def get_size(start_path):
    """Returns total size (in MB) of a folder or file."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except FileNotFoundError:
                continue
    return total_size / (1024 * 1024)  # Convert to MB

output_file = "directory_report.txt"

root_dir = "."  # current directory
folder_sizes = []
file_sizes = []

# Calculate size of each subfolder and each file
for root, dirs, files in os.walk(root_dir):
    # Avoid going into subdirs recursively for folder size summary
    for d in dirs:
        folder_path = os.path.join(root, d)
        folder_size = get_size(folder_path)
        folder_sizes.append((folder_path, folder_size))
    for f in files:
        file_path = os.path.join(root, f)
        try:
            size = os.path.getsize(file_path) / (1024 * 1024)
            file_sizes.append((file_path, size))
        except FileNotFoundError:
            continue
    break  # Only top-level folders and files for folder sizes

# Sort
folder_sizes.sort(key=lambda x: x[1], reverse=True)
file_sizes.sort(key=lambda x: x[1], reverse=True)

# Get total project size
total_project_size = get_size(root_dir)

# Write results
with open(output_file, "w") as f:
    f.write(f"Total project size: {total_project_size:.2f} MB\n\n")

    f.write("Folder Sizes (Top-level):\n")
    for path, size in folder_sizes:
        f.write(f"{path}: {size:.2f} MB\n")
    
    f.write("\nFile Sizes (All, sorted):\n")
    for path, size in file_sizes:
        f.write(f"{path}: {size:.2f} MB\n")

print(f"Full directory report saved to '{output_file}'")
