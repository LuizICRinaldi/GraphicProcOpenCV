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

SPLIT_PATH = PROJECT_ROOT / "results" / "data_splits.csv"

OUTPUT_DIR = PROJECT_ROOT / "results"


# ==========================================
# 2. CARREGAR CONJUNTO DE TESTE
# ==========================================

df = pd.read_csv(SPLIT_PATH)

test_df = df[df["split"] == "test"].copy()

print("Imagens no conjunto de teste:", len(test_df))

if test_df.empty:
    raise ValueError("Conjunto de teste vazio!")


# ==========================================
# 3. CALCULAR MÉTRICAS
# ==========================================

results = []

for _, row in test_df.iterrows():

    noisy_path = PROJECT_ROOT / row["noisy_path"]
    gt_path = PROJECT_ROOT / row["gt_path"]

    noisy = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)
    ground_truth = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

    if noisy is None or ground_truth is None:
        raise ValueError(f"Erro de leitura: {row['folder']}")

    if noisy.shape != ground_truth.shape:
        raise ValueError(f"Dimensões diferentes: {row['folder']}")

    # Converter BGR para RGB
    noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    ground_truth = cv2.cvtColor(
        ground_truth,
        cv2.COLOR_BGR2RGB
    )

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
    mae = np.mean(
        np.abs(
            noisy.astype(np.float32)
            - ground_truth.astype(np.float32)
        )
    )

    results.append({
        "folder": row["folder"],
        "brightness": row["brightness"],
        "PSNR": psnr,
        "SSIM": ssim,
        "MAE": mae
    })


# ==========================================
# 4. SALVAR RESULTADOS INDIVIDUAIS
# ==========================================

results_df = pd.DataFrame(results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    OUTPUT_DIR / "baseline_test_per_image.csv",
    index=False
)


# ==========================================
# 5. RESULTADOS AGREGADOS
# ==========================================

summary = []

groups = [
    ("ALL", results_df),
    *list(results_df.groupby("brightness"))
]

for name, group in groups:

    summary.append({
        "group": name,
        "images": len(group),
        "PSNR_mean": group["PSNR"].mean(),
        "PSNR_std": group["PSNR"].std(),
        "SSIM_mean": group["SSIM"].mean(),
        "SSIM_std": group["SSIM"].std(),
        "MAE_mean": group["MAE"].mean(),
        "MAE_std": group["MAE"].std()
    })


summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_DIR / "baseline_test_summary.csv",
    index=False
)


# ==========================================
# 6. EXIBIR RESULTADOS
# ==========================================

print("\n===== BASELINE DO TESTE =====")

print(
    summary_df.round(4).to_string(index=False)
)

print("\nArquivos gerados:")
print("baseline_test_per_image.csv")
print("baseline_test_summary.csv")