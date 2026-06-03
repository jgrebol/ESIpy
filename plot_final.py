import os
import pickle
import matplotlib.pyplot as plt

base_dir = "/home/joan/PycharmProjects/ESIpy/joan"

PARTITION_COLORS = {
    'MULLIKEN': 'red', 'LOWDIN': 'yellow', 'META-LOWDIN': 'limegreen', 
    'NAO': 'orange', 'IAO': 'blue', 'IAO-AUTOSAD': 'darkblue', 
    'QTAIM': 'black', 'IAO-EFFAO-GROSS': 'darkorange', 
    'IAO-EFFAO-LOWDIN': 'yellow', 'IAO-EFFAO-ML': 'limegreen', 
    'IAO-EFFAO-META-LOWDIN': 'limegreen',
    'IAO-EFFAO-SYMMETRIC': 'cyan'
}

# For parametrics we need a colormap
from matplotlib import cm
fpiao_colors = cm.get_cmap('plasma', 8)
dfpiao_colors = cm.get_cmap('viridis', 6)

def get_color(part, fam):
    part_upper = part.upper()
    if part_upper in PARTITION_COLORS:
        return PARTITION_COLORS[part_upper]
    elif 'FPIAO' in part_upper and fam == 'FPIAO':
        try:
            val = float(part_upper.replace('FPIAO', ''))
            # values are 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 (7 values)
            idx = int((val - 0.5) / 0.25)
            return fpiao_colors(idx)
        except:
            return 'grey'
    elif 'DFPIAO' in part_upper and fam == 'DFPIAO':
        try:
            val = float(part_upper.replace('DFPIAO', ''))
            # values are 0.3, 0.4, 0.5, 0.6, 0.7 (5 values)
            idx = int((val - 0.3) / 0.1)
            return dfpiao_colors(idx)
        except:
            return 'grey'
    return 'grey'

def plot_system(sys_name, pkl_file, out_prefix):
    if not os.path.exists(pkl_file):
        print(f"Skipping {sys_name}, file not found: {pkl_file}")
        return
        
    with open(pkl_file, 'rb') as f:
        d, e, data = pickle.load(f)
        
    for fam in data:
        plt.figure(figsize=(10, 6))
        for part in data[fam]:
            if len(data[fam][part]) > 0:
                color = get_color(part, fam)
                label = part.upper()
                plt.plot(d, data[fam][part], '-', linewidth=2.5, color=color, label=label)
                
        plt.xlabel('Distance (Å)', fontsize=14)
        plt.ylabel('DI', fontsize=14)
        plt.title(f'{sys_name} - {fam} Family', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        out_path = f"/home/joan/.gemini/antigravity-cli/brain/e97fce9c-463e-4f5b-8f3b-7cb078d54c5d/{out_prefix}_{fam}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

plot_system('LiH (Ground State)', os.path.join(base_dir, 'LiH/GS/chk/LiH-GS_data.pkl'), 'LiH_GS')
plot_system('LiH (Excited State)', os.path.join(base_dir, 'LiH/ES/chk/LiH-ES_data.pkl'), 'LiH_ES')
plot_system('LiF', os.path.join(base_dir, 'HARPOON/LiF/chk/LiF_data.pkl'), 'LiF')

