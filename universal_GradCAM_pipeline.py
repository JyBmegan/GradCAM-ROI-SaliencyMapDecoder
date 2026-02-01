import os
import glob
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d



MODEL_DIR = './All_Models_Collection'
INPUT_IMAGE_ROOT = './input_images' 
OUTPUT_ROOT = './Results'

# 开关
SAVE_IMAGES = True       # 保存纯净 Grad-CAM 热力图
SAVE_DATAPOINTS = True   # 保存矩阵数据
SAVE_TREND_PLOTS = True  # [新增] 保存趋势分析图

# ROI 
ROIS = {
    'Eyes':  {'x': (52, 171), 'y': (70, 99),   'color': 'cyan'},
    'Nose':  {'x': (52, 171), 'y': (120, 149), 'color': 'gold'}, 
    'Mouth': {'x': (52, 171), 'y': (169, 199), 'color': 'red'}  
}


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(in_channels // reduction, 1)
        self.fc1 = nn.Linear(in_channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 4:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
        else:
            b, c = x.size()
            y = x
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        y = self.sigmoid(y)
        if x.dim() == 4:
            y = y.view(b, c, 1, 1)
            return x * y.expand_as(x)
        else:
            return x * y

class UniversalSEAlexNet(nn.Module):  
    def __init__(self, num_classes=8, se_pos=None, reduction=4):
        super(UniversalSEAlexNet, self).__init__()
        pos_str = str(se_pos)
        layers = []
        layers += [nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2)]
        if pos_str == '1': layers.append(SEBlock(64, reduction))
        layers += [nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2)]
        if pos_str == '2': layers.append(SEBlock(192, reduction))
        layers += [nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=False)]
        if pos_str == '3': layers.append(SEBlock(384, reduction))
        layers += [nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False)]
        if pos_str == '4': layers.append(SEBlock(256, reduction))
        layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2)]
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.se_fc1 = SEBlock(4096, reduction) if pos_str in ['L1', 'Location1'] else None
        self.se_fc2 = SEBlock(4096, reduction) if pos_str in ['L2', 'Location2'] else None
        self.se_fc3 = SEBlock(4096, reduction) if pos_str in ['L3', 'Location3'] else None

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x), inplace=False)
        if self.se_fc1: x = self.se_fc1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x), inplace=False)
        if self.se_fc2: x = self.se_fc2(x)
        if self.se_fc3: x = self.se_fc3(x)
        x = self.fc3(x)
        return x

class UniversalVGG16(nn.Module):
    def __init__(self, num_classes=8):
        super(UniversalVGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=False), nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ================= 3. 工具函数 =================

def parse_model_config(filename):
    config = {
        'filename': filename, 'reduction': 4, 'pos': None, 'base_type': 'Unknown', 'arch': 'AlexNet' 
    }
    if 'VGG' in filename or 'vgg' in filename: config['arch'] = 'VGG16'
    r_match = re.search(r'_R(\d+)', filename)
    if r_match: config['reduction'] = int(r_match.group(1))
    if 'FaceBased' in filename: config['base_type'] = 'FaceBased'
    if 'ObjectBased' in filename: config['base_type'] = 'ObjectBased'
    if 'Baseline' in filename: config['pos'] = None
    elif 'SE-Conv-L' in filename:
        p_match = re.search(r'SE-Conv-L(\d+)', filename)
        if p_match: config['pos'] = int(p_match.group(1)) 
    elif 'SE-FC-L' in filename:
        p_match = re.search(r'SE-FC-L(\d+)', filename)
        if p_match: config['pos'] = f"L{p_match.group(1)}" 
    return config

def preprocess_image(pil_img):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return preprocess(pil_img).unsqueeze(0)

def get_last_conv_layer(model):
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

def resize_cam_tensor(cam_np, target_size=(224, 224)):
    cam_tensor = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0) 
    cam_resized = F.interpolate(cam_tensor, size=target_size, mode='bilinear', align_corners=False)
    return cam_resized.squeeze().numpy()

def calculate_roi_from_cam(cam_matrix):
    total = np.sum(cam_matrix)
    if total == 0: return {k:0.0 for k in ROIS}
    res = {}
    for name, coords in ROIS.items():
        xs, xe = min(coords['x']), max(coords['x']) + 1
        ys, ye = min(coords['y']), max(coords['y']) + 1
        res[name] = np.sum(cam_matrix[ys:ye, xs:xe]) / total
    return res

def save_pure_heatmap(cam_matrix, original_img, save_path):
    heatmap = cm.jet(cam_matrix)[..., :3] # 0-1 float
    orig_resized = original_img.resize((224, 224))
    orig_np = np.array(orig_resized).astype(float) / 255.0
    overlay = (heatmap * 0.5 + orig_np * 0.5)
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(save_path)

def save_datapoint(cam_matrix, save_path):
    np.savetxt(save_path, cam_matrix, fmt='%.6e')


def save_trend_visualization(cam_matrix, save_path):
    """
    生成垂直分布趋势图 (Vertical Profile)
    X轴: 激活强度 (累加)
    Y轴: 像素行 (0-224, 倒置以匹配人脸高度)
    背景: 叠加 ROI 区域颜色
    """

    row_sums = np.sum(cam_matrix, axis=1)

    sigma = 3
    row_sums_smooth = gaussian_filter1d(row_sums, sigma=sigma)
    
    plt.figure(figsize=(5, 6))
    y_axis = np.arange(len(row_sums))
    
    plt.plot(row_sums_smooth, y_axis, color='black', linewidth=1.5, label='Saliency Trend')
    plt.fill_betweenx(y_axis, 0, row_sums_smooth, color='gray', alpha=0.1)
    

    for name, data in ROIS.items():
        y_start, y_end = min(data['y']), max(data['y'])
        plt.axhspan(y_start, y_end, color=data['color'], alpha=0.3, label=name)
    
    plt.gca().invert_yaxis() 
    plt.title("Vertical Saliency Distribution")
    plt.xlabel("Accumulated Activation")
    plt.ylabel("Pixel Height (Y)")
    plt.legend(loc='upper right')
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def calculate_trend_stats(cam_matrix):
    """计算分布的统计特征"""
    row_sums = np.sum(cam_matrix, axis=1)
    
    # Peak Y
    peak_y = np.argmax(row_sums)
    peak_val = np.max(row_sums)
    
    # Lowest Y
    lowest_y = np.argmin(row_sums)
    
    peak_region = 'Background'
    for name, data in ROIS.items():
        y_start, y_end = min(data['y']), max(data['y'])
        if y_start <= peak_y <= y_end:
            peak_region = name
            break
            
    return {
        'Peak_Y': peak_y,
        'Peak_Value': peak_val,
        'Lowest_Y': lowest_y,
        'Peak_Region': peak_region
    }

def generate_summary_report(csv_path, output_dir):
    try:
        df = pd.read_csv(csv_path)
        summary_mean = df.groupby('Condition')[['Eyes_PoS', 'Nose_PoS', 'Mouth_PoS']].mean()
        summary_std = df.groupby('Condition')[['Eyes_PoS', 'Nose_PoS', 'Mouth_PoS']].std()
        
        order = ['Full', 'E', 'M', 'N']
        summary_mean = summary_mean.reindex(order)
        summary_std = summary_std.reindex(order)
        
        stats_path = os.path.join(output_dir, 'Summary_Statistics.csv')
        summary_export = summary_mean.copy()
        for col in summary_export.columns:
            summary_export[f'{col}_Std'] = summary_std[col]
        summary_export.to_csv(stats_path)
        print(f"   [Stats] Data saved to: {stats_path}")

        plot_path = os.path.join(output_dir, 'Summary_Chart.png')
        plt.style.use('ggplot')
        ax = summary_mean.plot(kind='bar', yerr=summary_std, capsize=4, 
                               figsize=(10, 6), width=0.8,
                               color=['cyan', 'gold', 'red'], 
                               edgecolor='black', alpha=0.8)
        
        plt.title('Average ROI Saliency by Condition', fontsize=14)
        plt.ylabel('Mean PoS', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"   [Chart] Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"   [Report Error] {e}")


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for subdir in ['GradCAM_Images', 'Datapoints', 'Trend_Plots']:
        if not os.path.exists(os.path.join(OUTPUT_ROOT, subdir)):
            os.makedirs(os.path.join(OUTPUT_ROOT, subdir))
    
    model_files = glob.glob(os.path.join(MODEL_DIR, '*.pth'))
    print(f"Found {len(model_files)} models in {MODEL_DIR}")
    
    gradcam_results = []
    trend_results = [] 
    
    for model_path in model_files:
        fname = os.path.basename(model_path)
        model_name_clean = fname.replace('.pth', '')
        print(f"\n>>> Processing Model: {fname}")
        
        config = parse_model_config(fname)
        
        if config['arch'] == 'VGG16':
            model = UniversalVGG16(num_classes=8)
            model.to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            except Exception: continue
        else:
            ratios_to_try = [config['reduction']] + [4, 16, 32, 8, 2]
            ratios_to_try = sorted(set(ratios_to_try), key=ratios_to_try.index)
            model = None
            for r in ratios_to_try:
                try:
                    temp_model = UniversalSEAlexNet(num_classes=8, se_pos=config['pos'], reduction=r)
                    temp_model.to(device)
                    temp_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                    model = temp_model
                    break 
                except RuntimeError as e:
                    if "size mismatch" in str(e): continue
                    else: break
            if model is None: continue

        model.eval()
        
        grads = []
        feats = []
        def grad_hook(_, grad_in, grad_out):
            grads.append(grad_out[0].detach())
        def feat_hook(_, input, output):
            feats.append(output.detach())
        
        last_conv = get_last_conv_layer(model)
        if last_conv:
            last_conv.register_forward_hook(feat_hook)
            last_conv.register_full_backward_hook(grad_hook)
        
        for cond in ['Full', 'E', 'M', 'N']:
            img_dir = os.path.join(INPUT_IMAGE_ROOT, cond)
            if not os.path.exists(img_dir): continue
            
            # 准备路径
            img_save_dir = os.path.join(OUTPUT_ROOT, 'GradCAM_Images', model_name_clean, cond)
            data_save_dir = os.path.join(OUTPUT_ROOT, 'Datapoints', model_name_clean, cond)
            trend_save_dir = os.path.join(OUTPUT_ROOT, 'Trend_Plots', model_name_clean, cond) # [新增]
            
            if SAVE_IMAGES and not os.path.exists(img_save_dir): os.makedirs(img_save_dir)
            if SAVE_DATAPOINTS and not os.path.exists(data_save_dir): os.makedirs(data_save_dir)
            if SAVE_TREND_PLOTS and not os.path.exists(trend_save_dir): os.makedirs(trend_save_dir)

            img_files = sorted(os.listdir(img_dir))
            for img_name in img_files:
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')): continue
                img_id_match = re.search(r'(\d+)', img_name)
                img_id = int(img_id_match.group(1)) if img_id_match else 0
                
                try:
                    pil_img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
                    input_tensor = preprocess_image(pil_img).to(device)
                except: continue

                grads = [] 
                feats = []
                output = model(input_tensor)
                target_class = torch.argmax(output)
                model.zero_grad()
                output[0, target_class].backward()
                
                if not grads or not feats: continue
                
                gradient = grads[0].cpu().numpy()[0]
                feature = feats[0].cpu().numpy()[0]
                weights = np.mean(gradient, axis=(1, 2))
                cam = np.zeros(feature.shape[1:], dtype=np.float32)
                for i, w in enumerate(weights):
                    cam += w * feature[i, :, :]
                
                cam = np.maximum(cam, 0)
                cam = resize_cam_tensor(cam, (224, 224))
                if np.max(cam) != 0:
                    cam_norm = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
                else:
                    cam_norm = cam

                if SAVE_IMAGES:
                    save_path = os.path.join(img_save_dir, f"{img_id}.jpg")
                    save_pure_heatmap(cam_norm, pil_img, save_path)
                
                if SAVE_DATAPOINTS:
                    save_path = os.path.join(data_save_dir, f"{img_id}.txt")
                    save_datapoint(cam_norm, save_path)
                
                if SAVE_TREND_PLOTS:
                    save_path = os.path.join(trend_save_dir, f"{img_id}_trend.jpg")
                    save_trend_visualization(cam_norm, save_path)
                    
                    t_stats = calculate_trend_stats(cam_norm)
                    t_row = {
                        'ModelFile': fname,
                        'Condition': cond,
                        'ImageID': img_id,
                        'Reduction': config['reduction'],
                        'Peak_Y': t_stats['Peak_Y'],
                        'Peak_Value': t_stats['Peak_Value'],
                        'Lowest_Y': t_stats['Lowest_Y'],
                        'Peak_Region': t_stats['Peak_Region']
                    }
                    trend_results.append(t_row)
                
                roi_data = calculate_roi_from_cam(cam_norm)
                row = {
                    'ModelFile': fname,
                    'Arch': config['arch'],
                    'Location': config['pos'] if config['pos'] else 'Baseline',
                    'BaseType': config['base_type'],
                    'Reduction': config['reduction'],
                    'Condition': cond,
                    'ImageID': img_id,
                    'Eyes_PoS': roi_data['Eyes'],
                    'Nose_PoS': roi_data['Nose'],
                    'Mouth_PoS': roi_data['Mouth']
                }
                gradcam_results.append(row)

    if gradcam_results:
        df = pd.DataFrame(gradcam_results)
        df.sort_values(by=['Arch', 'Location', 'Reduction', 'BaseType', 'Condition', 'ImageID'], inplace=True)
        save_path = os.path.join(OUTPUT_ROOT, 'GradCAM_Results.csv')
        df.to_csv(save_path, index=False)
        print(f"Done! Processed {len(df)} images.")
        
        generate_summary_report(save_path, OUTPUT_ROOT)
        
        if trend_results:
            df_trend = pd.DataFrame(trend_results)
            df_trend.sort_values(by=['ModelFile', 'Condition', 'ImageID'], inplace=True)
            trend_path = os.path.join(OUTPUT_ROOT, 'ROI_Trend_Stats.csv')
            df_trend.to_csv(trend_path, index=False)
            print(f"   [Trend Stats] Saved to: {trend_path}")
            
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()