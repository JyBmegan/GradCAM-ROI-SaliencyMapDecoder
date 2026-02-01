import os
import shutil
import re


PATH_GRAD_CAM = '/Volumes/ZX2 1TB/se-alexnet/GradCAMHeatMap'
PATH_CONV = '/Volumes/ZX2 1TB/Results/ConvResults'
DEST_DIR = './All_Models_Collection'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def copy_model(src, new_name):
    dst = os.path.join(DEST_DIR, new_name)
    print(f"Copying: {new_name}")
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"Error copying {src}: {e}")

# Insert in Fc Layer: SELocate-x
def organize_fc_models():
    print("\n--- Processing FC Models ---")
    root = PATH_GRAD_CAM
    if not os.path.exists(root):
        print(f"Path not found: {root}")
        return

    for item in os.listdir(root):
        if item.startswith('SELocate-'): 
            loc_path = os.path.join(root, item)
            # Insert Location: SELocate-1 -> L1
            try:
                loc_id = item.split('-')[-1]
            except: continue
            
            if not os.path.isdir(loc_path): continue

            # BaseType (FaceBased / ObjectBased)
            for base_type in os.listdir(loc_path):
                base_path = os.path.join(loc_path, base_type)
                if not os.path.isdir(base_path): continue
                
                # Ratio (squeeze-2, squeeze-4...)
                for sq_folder in os.listdir(base_path):
                    if sq_folder.startswith('squeeze-'):
                        ratio = sq_folder.split('-')[-1]
                        
                        #  .pth 
                        sq_path = os.path.join(base_path, sq_folder)
                        pth_files = [f for f in os.listdir(sq_path) if f.endswith('.pth')]
                        
                        if pth_files:
                            src_file = os.path.join(sq_path, pth_files[0])
                            # Rename: SE-FC-L1_FaceBased_R2.pth
                            new_name = f"SE-FC-L{loc_id}_{base_type}_R{ratio}.pth"
                            copy_model(src_file, new_name)

#  Baseline(Raw...)
def organize_baseline_models():
    print("\n--- Processing Baseline Models ---")
    root = PATH_GRAD_CAM 
    if not os.path.exists(root): return
    for model_folder in os.listdir(root):
        # Key: Raw/Alex/VGG
        if ('Raw' in model_folder or 'Alex' in model_folder or 'VGG' in model_folder) and not model_folder.startswith('SELocate'):
            
            model_path_root = os.path.join(root, model_folder)
            if not os.path.isdir(model_path_root): continue

            # BaseType
            for base_type in os.listdir(model_path_root):
                base_path = os.path.join(model_path_root, base_type)
                if not os.path.isdir(base_path): continue
                
                #  .pth
                pth_files = [f for f in os.listdir(base_path) if f.endswith('.pth')]
                
                if pth_files:
                    src_file = os.path.join(base_path, pth_files[0])
                    filename = pth_files[0]
                    
                    if 'vgg' in model_folder.lower() or 'vgg' in filename.lower():
                        model_name = 'VGG16'
                    elif 'alex' in model_folder.lower() or 'alex' in filename.lower():
                        model_name = 'AlexNet'
                    else:
                        model_name = model_folder.replace('Raw', '') # Fallback

                    # Rename: Baseline_AlexNet_FaceBased.pth
                    new_name = f"Baseline_{model_name}_{base_type}.pth"
                    copy_model(src_file, new_name)

# Insert Conv (SeC...)
def organize_conv_models():
    print("\n--- Processing Conv Models ---")
    root = PATH_CONV
    if not os.path.exists(root): 
        print(f"Conv path not found: {root}")
        return

    for folder in os.listdir(root):
        # SeC1_FaceBased_squeeze2
        if folder.startswith('SeC'):
            parts = folder.split('_') 
            if len(parts) >= 3:
                loc_raw = parts[0]   # SeC1
                base_type = parts[1] # FaceBased
                ratio_raw = parts[2] # squeeze2
                
                # Location
                loc_id = loc_raw.replace('SeC', 'L') # SeC1 -> L1
                ratio_match = re.search(r'\d+', ratio_raw)
                if ratio_match:
                    ratio = ratio_match.group()
                    
                    model_path = os.path.join(root, folder, 'AlexNet.pth')
                    #  pth
                    if not os.path.exists(model_path):
                        files = [f for f in os.listdir(os.path.join(root, folder)) if f.endswith('.pth')]
                        if files: model_path = os.path.join(root, folder, files[0])
                    
                    if os.path.exists(model_path):
                        # Rename: SE-Conv-L1_FaceBased_R2.pth
                        new_name = f"SE-Conv-{loc_id}_{base_type}_R{ratio}.pth"
                        copy_model(model_path, new_name)

if __name__ == "__main__":
    ensure_dir(DEST_DIR)
    
    organize_fc_models()
    organize_baseline_models()
    organize_conv_models()
    
    print(f"Done! Save to {DEST_DIR}")
