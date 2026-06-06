import pandas as pd
import os

def merge_csv_files():
    # Daftar file yang akan digabung (sudah ditambahkan all_models_summary.csv)
    files_to_merge = [
        'results/results_final.csv', 
        'results/results_v2.csv', 
        'results/ablation_results.csv',
        'results/all_models_summary.csv'
    ]
    
    dfs = []
    for file in files_to_merge:
        if os.path.exists(file):
            df = pd.read_csv(file)
            
            # Standarisasi kolom untuk all_models_summary.csv
            if 'Model Name' in df.columns:
                df.rename(columns={
                    'Model Name': 'method',
                    'Best Dice': 'dice',
                    'Best mIoU': 'iou',
                    'Best HD95': 'hd95'
                }, inplace=True)
                
            # Standarisasi kolom untuk ablation_results.csv
            if 'config' in df.columns:
                df.rename(columns={'config': 'method'}, inplace=True)
                
            dfs.append(df)
        else:
            print(f"Peringatan: File {file} tidak ditemukan.")

    if not dfs:
        print("Tidak ada data CSV yang bisa digabung.")
        return

    # Gabungkan semua dataframe
    merged_df = pd.concat(dfs, ignore_index=True)

    # Hapus duplikat berdasarkan nama model (method) dan skor dice
    merged_df.drop_duplicates(subset=['method', 'dice'], keep='first', inplace=True)

    # Urutkan berdasarkan nilai Dice tertinggi ke terendah
    merged_df.sort_values(by='dice', ascending=False, inplace=True)

    # Simpan ke CSV master
    output_path = 'results/MASTER_MERGED_RESULTS.csv'
    merged_df.to_csv(output_path, index=False)
    
    print(f"Berhasil menggabungkan CSV ke '{output_path}'")
    print("\n--- Preview Hasil Gabungan (Diurutkan berdasarkan Dice tertinggi) ---")
    # Tampilkan kolom penting saja untuk preview
    print(merged_df[['method', 'dice', 'iou', 'hd95']].to_string(index=False))

if __name__ == "__main__":
    merge_csv_files()