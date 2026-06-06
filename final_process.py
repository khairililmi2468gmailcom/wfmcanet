import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob

def process_everything():
    print("--- 1. Mengekstrak Data dari Log Files ---")
    log_files = glob.glob('*.log')
    summary_list = []
    
    # Plotting Setup
    plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(1, 3, 1); ax1.set_title('Loss')
    ax2 = plt.subplot(1, 3, 2); ax2.set_title('Dice Score')
    ax3 = plt.subplot(1, 3, 3); ax3.set_title('mIoU')

    for file in sorted(log_files):
        model_name = file.replace('.log', '')
        epochs, losses, dices, mious, hd95s = [], [], [], [], []
        
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'loss=' in line and 'Dice=' in line:
                    try:
                        ep_match = re.search(r'Ep\s+(\d+)/', line)
                        loss_match = re.search(r'loss=([0-9.]+)', line)
                        dice_match = re.search(r'Dice=([0-9.]+)', line)
                        miou_match = re.search(r'mIoU=([0-9.]+)', line)
                        hd95_match = re.search(r'HD95=([0-9.]+)px', line)
                        
                        if ep_match and loss_match and dice_match and miou_match:
                            epochs.append(int(ep_match.group(1)))
                            losses.append(float(loss_match.group(1)))
                            dices.append(float(dice_match.group(1)))
                            mious.append(float(miou_match.group(1)))
                            if hd95_match: hd95s.append(float(hd95_match.group(1)))
                    except: continue
        
        if epochs:
            summary_list.append({
                'method': model_name,
                'dice': max(dices),
                'iou': max(mious),
                'hd95': min(hd95s) if hd95s else None,
                'epochs': len(epochs)
            })
            ax1.plot(epochs, losses, label=model_name)
            ax2.plot(epochs, dices, label=model_name)
            ax3.plot(epochs, mious, label=model_name)

    # Simpan Grafik
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/combined_training_curves.png', dpi=300)
    print("Grafik berhasil dibuat di results/combined_training_curves.png")

    print("\n--- 2. Menggabungkan Semua CSV (Merge) ---")
    # Daftar CSV yang ada di folder results
    csv_files = [
        'results/results_final.csv', 
        'results/results_v2.csv', 
        'results/ablation_results.csv'
    ]
    
    all_dfs = []
    
    # Masukkan data dari Log yang baru diproses
    if summary_list:
        all_dfs.append(pd.DataFrame(summary_list))

    # Masukkan data dari CSV lama
    for f_path in csv_files:
        if os.path.exists(f_path):
            df = pd.read_csv(f_path)
            # Standarisasi Nama Kolom
            if 'config' in df.columns: df.rename(columns={'config': 'method'}, inplace=True)
            if 'Dice' in df.columns: df.rename(columns={'Dice': 'dice'}, inplace=True)
            if 'mIoU' in df.columns: df.rename(columns={'mIoU': 'iou'}, inplace=True)
            all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Hapus duplikat dan bersihkan
        final_df.drop_duplicates(subset=['method', 'dice'], keep='first', inplace=True)
        # Urutkan berdasarkan Dice terbaik
        final_df = final_df.sort_values(by='dice', ascending=False)
        
        final_df.to_csv('results/MASTER_MERGED_RESULTS.csv', index=False)
        print("MASTER_MERGED_RESULTS.csv berhasil dibuat.")
        print(final_df[['method', 'dice', 'iou']].head(15).to_string(index=False))

if __name__ == "__main__":
    process_everything()