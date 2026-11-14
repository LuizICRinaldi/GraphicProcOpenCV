from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ==========================================
# 1. CAMINHOS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "results" / "pairs_ip.csv"

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CARREGAR CATÁLOGO
# ==========================================

df = pd.read_csv(CSV_PATH)

results = []


# ==========================================
# 3. CALCULAR MÉTRICAS
# ==========================================

for _, row in df.iterrows():

    noisy_path = PROJECT_ROOT / row["noisy_path"]
    gt_path = PROJECT_ROOT / row["gt_path"]

    noisy = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)
    ground_truth = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

    if noisy is None or ground_truth is None:
        print(f"Erro ao carregar: {row['folder']}")
        continue

    if noisy.shape != ground_truth.shape:
        print(f"Dimensões diferentes: {row['folder']}")
        continue

    # Converter para RGB
    noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    ground_truth = cv2.cvtColor(
        ground_truth,
        cv2.COLOR_BGR2RGB
    )

    # Converter para float para calcular MAE
    noisy_float = noisy.astype(np.float32)
    gt_float = ground_truth.astype(np.float32)

    # PSNR
    psnr = peak_signal_noise_ratio(
        ground_truth,
        noisy,
        data_range=255
    )

    # SSIM
    ssim = structural_similarity(
        ground_truth,
        noisy,
        channel_axis=-1,
        data_range=255
    )

    # MAE
    mae = np.mean(np.abs(noisy_float - gt_float))

    # Condição de iluminação
    brightness = row["folder"].split("_")[-1]

    results.append({
        "folder": row["folder"],
        "brightness": brightness,
        "PSNR": psnr,
        "SSIM": ssim,
        "MAE": mae
    })

    print(
        f"{row['folder']} | "
        f"PSNR: {psnr:.2f} | "
        f"SSIM: {ssim:.4f} | "
        f"MAE: {mae:.2f}"
    )


# ==========================================
# 4. SALVAR RESULTADOS INDIVIDUAIS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "baseline_per_image.csv",
    index=False
)


# ==========================================
# 5. CALCULAR RESULTADOS AGREGADOS
# ==========================================

summary_rows = []

# Resultado geral
summary_rows.append({
    "group": "ALL",
    "images": len(results_df),
    "PSNR_mean": results_df["PSNR"].mean(),
    "SSIM_mean": results_df["SSIM"].mean(),
    "MAE_mean": results_df["MAE"].mean()
})

# Resultados por iluminação
for condition, group in results_df.groupby("brightness"):

    summary_rows.append({
        "group": condition,
        "images": len(group),
        "PSNR_mean": group["PSNR"].mean(),
        "SSIM_mean": group["SSIM"].mean(),
        "MAE_mean": group["MAE"].mean()
    })


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "baseline_summary.csv",
    index=False
)


# ==========================================
# 6. EXIBIR RESULTADOS
# ==========================================

print("\n===== BASELINE FINAL =====")

print(summary_df.to_string(index=False))

print("\nArquivos gerados:")
print("results/baseline_per_image.csv")
print("results/baseline_summary.csv")