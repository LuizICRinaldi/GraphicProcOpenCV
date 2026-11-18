from pathlib import Path
import json
import time

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from denoising_methods import (
    apply_method,
    calculate_metrics
)


# ==========================================
# 1. CAMINHOS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_PATH = ROOT / "results" / "data_splits.csv"

CONFIG_PATH = (
    ROOT
    / "results"
    / "method_comparison"
    / "validation_best_configs.json"
)

OUTPUT = ROOT / "results" / "final_test"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CARREGAR TESTE E CONFIGURAÇÕES
# ==========================================

df = pd.read_csv(SPLIT_PATH)

test_df = df[df["split"] == "test"].copy()

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    selection = json.load(file)

noise_variance = selection["noise_variance_bgr"]
best_configs = selection["selected_configs"]

if len(test_df) != 12:
    print(
        "Aviso: quantidade de imagens diferente das "
        "12 esperadas."
    )

print("Imagens do teste:", len(test_df))


# ==========================================
# 3. AVALIAR MÉTODOS
# ==========================================

methods_order = [
    "Gaussian",
    "Median",
    "Bilateral",
    "Adaptive",
    "Wavelet"
]

results = []

VISUAL_DIR = OUTPUT / "comparisons"
RESTORED_DIR = OUTPUT / "restored"

VISUAL_DIR.mkdir(parents=True, exist_ok=True)

for method in methods_order:
    (RESTORED_DIR / method.lower()).mkdir(
        parents=True,
        exist_ok=True
    )


for _, row in test_df.iterrows():

    noisy = cv2.imread(
        str(ROOT / row["noisy_path"]),
        cv2.IMREAD_COLOR
    )

    gt = cv2.imread(
        str(ROOT / row["gt_path"]),
        cv2.IMREAD_COLOR
    )

    if noisy is None or gt is None:
        raise ValueError(f"Erro de leitura: {row['folder']}")

    if noisy.shape != gt.shape:
        raise ValueError(f"Dimensões diferentes: {row['folder']}")

    # --------------------------------------
    # ORIGINAL SEM FILTRO
    # --------------------------------------

    original_metrics = calculate_metrics(noisy, gt)

    results.append({
        "folder": row["folder"],
        "brightness": row["brightness"],
        "method": "Original",
        **original_metrics,
        "time_ms": 0
    })

    outputs = {}

    # --------------------------------------
    # CINCO MÉTODOS DE RESTAURAÇÃO
    # --------------------------------------

    for method in methods_order:

        params = best_configs[method]["params"]

        start = time.perf_counter()

        restored = apply_method(
            noisy,
            method,
            params,
            noise_variance
        )

        elapsed = time.perf_counter() - start

        metrics = calculate_metrics(restored, gt)

        results.append({
            "folder": row["folder"],
            "brightness": row["brightness"],
            "method": method,
            **metrics,
            "time_ms": elapsed * 1000
        })

        outputs[method] = restored

        cv2.imwrite(
            str(
                RESTORED_DIR
                / method.lower()
                / f"{row['folder']}.png"
            ),
            restored
        )

    print(f"Teste concluído: {row['folder']}")

    # --------------------------------------
    # COMPARAÇÃO VISUAL
    # --------------------------------------

    noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(
        1, 7,
        figsize=(22, 5)
    )

    axes[0].imshow(noisy_rgb)
    axes[0].set_title("Original")

    for i, method in enumerate(methods_order, start=1):

        restored_rgb = cv2.cvtColor(
            outputs[method],
            cv2.COLOR_BGR2RGB
        )

        axes[i].imshow(restored_rgb)
        axes[i].set_title(method)

    axes[6].imshow(gt_rgb)
    axes[6].set_title("Ground Truth")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(row["folder"])
    plt.tight_layout()

    fig.savefig(
        VISUAL_DIR / f"{row['folder']}.png",
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# 4. SALVAR RESULTADOS INDIVIDUAIS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "final_per_image.csv",
    index=False
)


# ==========================================
# 5. CALCULAR RESULTADOS AGREGADOS
# ==========================================

summary = []

for method in results_df["method"].unique():

    method_df = results_df[
        results_df["method"] == method
    ]

    for condition in ["ALL", "L", "N"]:

        if condition == "ALL":
            group = method_df
        else:
            group = method_df[
                method_df["brightness"] == condition
            ]

        if group.empty:
            continue

        summary.append({
            "method": method,
            "group": condition,
            "images": len(group),
            "PSNR_mean": group["PSNR"].mean(),
            "PSNR_std": group["PSNR"].std(),
            "SSIM_mean": group["SSIM"].mean(),
            "SSIM_std": group["SSIM"].std(),
            "MAE_mean": group["MAE"].mean(),
            "MAE_std": group["MAE"].std(),
            "Edge_MAE_mean": group["Edge_MAE"].mean(),
            "time_ms_mean": group["time_ms"].mean()
        })


summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT / "final_summary.csv",
    index=False
)


# ==========================================
# 6. EXIBIR RESULTADOS
# ==========================================

print("\n===== AVALIAÇÃO FINAL =====")

print(
    summary_df[
        summary_df["group"] == "ALL"
    ].round(4).to_string(index=False)
)

print("\nArquivos gerados:")
print("final_per_image.csv")
print("final_summary.csv")
print("comparisons/")
print("restored/")