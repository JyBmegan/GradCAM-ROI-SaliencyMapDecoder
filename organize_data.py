import os
import shutil
import glob

SOURCE_ROOT = './ConvResults'
DEST_ROOT = './Data'

def normalize_folder_name(folder_name):
    parts = folder_name.split('-')
    if parts[0] == 'SeC' and len(parts) > 1 and parts[1].isdigit():
        new_prefix = f"SeC{parts[1]}"
        rest = "-".join(parts[2:])
        return f"{new_prefix}-{rest}"
    return folder_name

def main():
    if not os.path.exists(DEST_ROOT):
        os.makedirs(DEST_ROOT)

    count = 0
    exp_dirs = [f.path for f in os.scandir(SOURCE_ROOT) if f.is_dir()]
    
    for exp_dir in exp_dirs:
        datapoint_path = os.path.join(exp_dir, 'datapoint')
        if os.path.exists(datapoint_path):
            cond_folders = [f for f in os.scandir(datapoint_path) if f.is_dir()]
            for folder in cond_folders:
                std_name = normalize_folder_name(folder.name)
                src = folder.path
                dst = os.path.join(DEST_ROOT, std_name)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"[Organized] {folder.name} -> {std_name}")
                count += 1
    
    print(f"\n Organized {count} folders into {DEST_ROOT}")

if __name__ == "__main__":
    main()