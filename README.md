# Plastic Detection BYOL-style Self-Supervised Learning YOLOBackbone

# Plastic Detection — BYOL (YOLO10, YOLO11, and YOLOv12 backbone) → YOLOv10, YOLO11, YOLOv12 Fine-Tune

**Goal.** Self-supervised pretrain the YOLO backbone on your images to detect plastic using **BYOL**, then fine-tune a YOLOv10, YOLOv11, and YOLOv12 detector. The notebook also provides quick **representation diagnostics** (k-NN, t-SNE, PCA) and one-click visualizations. 

---

## What this notebook does
1. **Data prep**: Convert COCO → YOLO format with progress bars.
2. **BYOL pretraining (SSL)**: 
   - Detect pre-hook to grab FPN maps (P3/P4/P5) → global-pool + concat.  
   - Online/Target networks with EMA, projector/predictor heads.  
   - Cosine LR + warmup, param groups (no WD on norms/bias), grad clipping.
3. **Supervised fine-tune (Detection)**: YOLO10, YOLO11, and YOLOv12 with AdamW, cosine LR, EMA, `rect=True`, `cache="ram"`.
4. **Diagnostics (optional)**: k-NN confusion matrix, t-SNE, PCA of BYOL features, sample detection plot.

---

## Dataset
- Root (COCO style):  
  `/kaggle/input/plastic-coco/plastic_coco`  
- After conversion, YOLO folders are written under:  
  `/kaggle/working/plastic_ssl/0_yolo_split/{train,val,test}`

---

## Key toggles (edit in code)
- `MODEL = "yolo12n.yaml"`  – backbone/detector size.
- `SSL_EPOCHS = 75`          – BYOL epochs (sample config; raise for better reps).
- Detector `epochs=75`       – in `det.train(...)`.
- `KNN_VIZ = True`          – enable BYOL k-NN + t-SNE plots.

---

## Repro/Environment
- Seeds set (`42`) for NumPy & PyTorch.
- Notebook installs/uses a recent **ultralytics** build (YOLOv12).
- GPU recommended. If CUDA isn’t available, the code will still run in CPU (slower).

---

## Reading the results
- **Training logs**: BYOL loss (↓), alignment (↓), uniformity (↓) indicate healthier reps.
- **Detector metrics**: focus on **mAP@0.50:0.95**, **mAP@0.50**, **Precision/Recall**.
- **k-NN / t-SNE / PCA**: clusters and accuracy provide a quick sanity check of learned features.
- **Ablation Study on YOLO10, YOLO11, and YOLO12 based parameters like Learning Rate, Mosaic, Mixup and more

---

