from pathlib import Path
import matplotlib.pyplot as plt
import time

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

OUTPUT_DIR = PROJECT_ROOT / "results" / "gaussian"

RESTORED_DIR = OUTPUT_DIR / "restored"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"

RESTORED_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CARREGAR VALIDAÇÃO
# ==========================================

df = pd.read_csv(SPLIT_PATH)

validation_df = df[df["split"] == "validation"]

print("Imagens de validação:", len(validation_df))


# ==========================================
# 3. CONFIGURAÇÕES DO GAUSSIANO
# ==========================================

kernels = [3, 5, 7]
sigmas = [0.5, 1.0, 1.5]

results = []


# ==========================================
# 4. TESTAR CONFIGURAÇÕES
# ==========================================

for kernel in kernels:

    for sigma in sigmas:

        print(f"\nKernel: {kernel}x{kernel} | Sigma: {sigma}")

        for _, row in validation_df.iterrows():

            noisy_path = PROJECT_ROOT / row["noisy_path"]
            gt_path = PROJECT_ROOT / row["gt_path"]

            noisy = cv2.imread(
                str(noisy_path),
                cv2.IMREAD_COLOR
            )

            ground_truth = cv2.imread(
                str(gt_path),
                cv2.IMREAD_COLOR
            )

            if noisy is None or ground_truth is None:
                raise ValueError(f"Erro na leitura: {row['folder']}")

            if noisy.shape != ground_truth.shape:
                raise ValueError(f"Dimensões diferentes: {row['folder']}")

            # Medir somente o tempo do filtro
            start = time.perf_counter()

            restored = cv2.GaussianBlur(
                noisy,
                (kernel, kernel),
                sigmaX=sigma
            )

            elapsed = time.perf_counter() - start

            # Calcular métricas
            psnr = peak_signal_noise_ratio(
                ground_truth,
                restored,
                data_range=255
            )

            ssim = structural_similarity(
                ground_truth,
                restored,
                channel_axis=-1,
                data_range=255
            )

            mae = np.mean(
                np.abs(
                    restored.astype(np.float32)
                    - ground_truth.astype(np.float32)
                )
            )

            results.append({
                "folder": row["folder"],
                "brightness": row["brightness"],
                "kernel": kernel,
                "sigma": sigma,
                "PSNR": psnr,
                "SSIM": ssim,
                "MAE": mae,
                "time_ms": elapsed * 1000
            })

            # ==========================================
            # 5. SALVAR IMAGENS E COMPARAÇÕES
            # ==========================================

            if kernel == 7 and sigma == 1.5:

                filename = row["folder"] + ".png"

                # Salvar imagem restaurada
                cv2.imwrite(
                    str(RESTORED_DIR / filename),
                    restored
                )

                # Converter BGR para RGB
                noisy_rgb = cv2.cvtColor(
                    noisy, cv2.COLOR_BGR2RGB
                )

                restored_rgb = cv2.cvtColor(
                    restored, cv2.COLOR_BGR2RGB
                )

                gt_rgb = cv2.cvtColor(
                    ground_truth, cv2.COLOR_BGR2RGB
                )

                # Criar comparação visual
                fig, axes = plt.subplots(
                    1, 3, figsize=(16, 6)
                )

                axes[0].imshow(noisy_rgb)
                axes[0].set_title("Noisy sRGB")

                axes[1].imshow(restored_rgb)
                axes[1].set_title("Gaussian 7x7 - Sigma 1.5")

                axes[2].imshow(gt_rgb)
                axes[2].set_title("Ground Truth")

                for ax in axes:
                    ax.axis("off")

                fig.suptitle(row["folder"])
                plt.tight_layout()

                # Salvar comparação
                fig.savefig(
                    str(COMPARISON_DIR / filename),
                    dpi=150,
                    bbox_inches="tight"
                )

                plt.close(fig)

                print(f"Comparação salva: {filename}")


# ==========================================
# 5. RESULTADOS INDIVIDUAIS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "gaussian_per_image.csv",
    index=False
)


# ==========================================
# 6. RESULTADOS AGREGADOS
# ==========================================

summary = (
    results_df
    .groupby(["kernel", "sigma"])
    .agg(
        images=("folder", "count"),
        PSNR_mean=("PSNR", "mean"),
        PSNR_std=("PSNR", "std"),
        SSIM_mean=("SSIM", "mean"),
        SSIM_std=("SSIM", "std"),
        MAE_mean=("MAE", "mean"),
        time_ms_mean=("time_ms", "mean")
    )
    .reset_index()
)

summary = summary.sort_values(
    by="SSIM_mean",
    ascending=False
)

summary.to_csv(
    OUTPUT_DIR / "gaussian_summary.csv",
    index=False
)


# ==========================================
# 7. EXIBIR RESULTADOS
# ==========================================

print("\n===== RESULTADOS GAUSSIANOS =====")

print(summary.round(4).to_string(index=False))

print("\nArquivos gerados:")
print("gaussian_per_image.csv")
print("gaussian_summary.csv")