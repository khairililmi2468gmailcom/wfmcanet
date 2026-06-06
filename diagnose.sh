echo "=== DATASET CHECK ===" && \
echo "ISIC train images:" && ls /mnt/gpu17/segilmi/data/ISIC2018/ISIC2018_Task1-2_Training_Input/*.jpg 2>/dev/null | wc -l && \
echo "ISIC train masks:" && ls /mnt/gpu17/segilmi/data/ISIC2018/ISIC2018_Task1_Training_GroundTruth/*.png 2>/dev/null | wc -l && \
echo "Kvasir images:" && ls /mnt/gpu17/segilmi/data/Kvasir-SEG/images/ | wc -l && \
echo "Kvasir masks jpg:" && ls /mnt/gpu17/segilmi/data/Kvasir-SEG/masks/*.jpg 2>/dev/null | wc -l && \
echo "Kvasir masks png:" && ls /mnt/gpu17/segilmi/data/Kvasir-SEG/masks/*.png 2>/dev/null | wc -l && \
echo "=== SAMPLE NAMES ===" && \
echo "First 3 Kvasir images:" && ls /mnt/gpu17/segilmi/data/Kvasir-SEG/images/ | head -3 && \
echo "First 3 Kvasir masks:" && ls /mnt/gpu17/segilmi/data/Kvasir-SEG/masks/ | head -3 && \
echo "=== SPLIT ===" && \
python3 -c "
import json, os
s = json.load(open('/mnt/gpu17/segilmi/data/isic2018_split.json'))
img, msk = s['train'][0]
print(f'Train={len(s[\"train\"])}, Test={len(s[\"test\"])}')
print(f'Image exists: {os.path.exists(img)} — {img}')
print(f'Mask  exists: {os.path.exists(msk)} — {msk}')
"
