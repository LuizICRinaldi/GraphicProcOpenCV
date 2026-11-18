from pathlib import Path
import json
import time

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from denoising_methods import (
    estimate_noise_variance,
    apply_method,
    calculate_metrics
)


# ==========================================
# 1. CAMINHOS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_PATH = ROOT / "results" / "data_splits.csv"

OUTPUT = ROOT / "results" / "method_comparison"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CARREGAR CONJUNTOS
# ==========================================

df = pd.read_csv(SPLIT_PATH)

train_df = df[df["split"] == "train"].copy()
validation_df = df[df["split"] == "validation"].copy()

if train_df.empty or validation_df.empty:
    raise ValueError("Treinamento ou validação vazios.")

print("Treinamento:", len(train_df))
print("Validação:", len(validation_df))


# ==========================================
# 3. ESTIMAR VARIÂNCIA DO RUÍDO
# ==========================================

noise_variance = estimate_noise_variance(
    train_df,
    ROOT
)

print("\nVariância estimada por canal BGR:")
print(noise_variance)


# ==========================================
# 4. CONFIGURAÇÕES EXPERIMENTAIS
# ==========================================

configs = [
    {
        "method": "Gaussian",
        "params": {"kernel": 7, "sigma": 1.5}
    },

    *[
        {
            "method": "Median",
            "params": {"ksize": k}
        }
        for k in [3, 5, 7]
    ],

    {
        "method": "Bilateral",
        "params": {
            "d": 5,
            "sigmaColor": 25,
            "sigmaSpace": 25
        }
    },
    {
        "method": "Bilateral",
        "params": {
            "d": 7,
            "sigmaColor": 50,
            "sigmaSpace": 50
        }
    },
    {
        "method": "Bilateral",
        "params": {
            "d": 9,
            "sigmaColor": 75,
            "sigmaSpace": 75
        }
    },

    *[
        {
            "method": "Adaptive",
            "params": {"window": w}
        }
        for w in [3, 5, 7]
    ],

    *[
        {
            "method": "Wavelet",
            "params": {
                "multiplier": m,
                "wavelet": "db2",
                "level": 2
            }
        }
        for m in [0.6, 0.8, 1.0]
    ]
]


# ==========================================
# 5. EXECUTAR EXPERIMENTOS NA VALIDAÇÃO
# ==========================================

results = []

# Baseline das imagens de validação
for _, row in validation_df.iterrows():

    noisy = cv2.imread(
        str(ROOT / row["noisy_path"]),
        cv2.IMREAD_COLOR
    )

    gt = cv2.imread(
        str(ROOT / row["gt_path"]),
        cv2.IMREAD_COLOR
    )

    metrics = calculate_metrics(noisy, gt)

    results.append({
        "folder": row["folder"],
        "brightness": row["brightness"],
        "method": "Original",
        "config_id": "baseline",
        "params": "baseline",
        **metrics,
        "time_ms": 0
    })

for index, config in enumerate(configs, start=1):

    method = config["method"]
    params = config["params"]

    config_id = json.dumps(params, sort_keys=True)

    print(
        f"\n[{index}/{len(configs)}] "
        f"{method} | {params}"
    )

    for _, row in validation_df.iterrows():

        noisy_path = ROOT / row["noisy_path"]
        gt_path = ROOT / row["gt_path"]

        noisy = cv2.imread(
            str(noisy_path),
            cv2.IMREAD_COLOR
        )

        gt = cv2.imread(
            str(gt_path),
            cv2.IMREAD_COLOR
        )

        if noisy is None or gt is None:
            raise ValueError(f"Erro na leitura: {row['folder']}")

        if noisy.shape != gt.shape:
            raise ValueError(f"Dimensões diferentes: {row['folder']}")

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
            "config_id": config_id,
            "params": config_id,
            **metrics,
            "time_ms": elapsed * 1000
        })

    print("Configuração concluída.")


# ==========================================
# 6. SALVAR RESULTADOS INDIVIDUAIS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "validation_per_image.csv",
    index=False
)


# ==========================================
# 7. RESULTADOS AGREGADOS
# ==========================================

summary = (
    results_df
    .groupby(["method", "config_id"])
    .agg(
        images=("folder", "count"),
        PSNR_mean=("PSNR", "mean"),
        PSNR_std=("PSNR", "std"),
        SSIM_mean=("SSIM", "mean"),
        SSIM_std=("SSIM", "std"),
        MAE_mean=("MAE", "mean"),
        Edge_MAE_mean=("Edge_MAE", "mean"),
        time_ms_mean=("time_ms", "mean")
    )
    .reset_index()
)

summary = summary.sort_values(
    ["method", "SSIM_mean"],
    ascending=[True, False]
)

summary.to_csv(
    OUTPUT / "validation_summary.csv",
    index=False
)

print("\n===== RESULTADOS DA VALIDAÇÃO =====")

print(
    summary.round(4).to_string(index=False)
)


# ==========================================
# 8. SELECIONAR UMA CONFIGURAÇÃO POR MÉTODO
# ==========================================

best_configs = {}

for method in [
    "Gaussian",
    "Median",
    "Bilateral",
    "Adaptive",
    "Wavelet"
]:

    method_results = summary[
        summary["method"] == method
    ]

    best = method_results.sort_values(
        "SSIM_mean",
        ascending=False
    ).iloc[0]

    best_configs[method] = {
        "params": json.loads(best["config_id"]),
        "SSIM_mean": float(best["SSIM_mean"]),
        "PSNR_mean": float(best["PSNR_mean"]),
        "MAE_mean": float(best["MAE_mean"])
    }

    print(
        f"\nSelecionado: {method} | "
        f"{best['config_id']} | "
        f"SSIM: {best['SSIM_mean']:.4f}"
    )


# Salvar configurações e estimativa de ruído
selection = {
    "noise_variance_bgr": noise_variance.tolist(),
    "selected_configs": best_configs
}

selection_path = OUTPUT / "validation_best_configs.json"

with open(selection_path, "w", encoding="utf-8") as file:
    json.dump(selection, file, indent=4)


# ==========================================
# 9. GERAR IMAGENS RESTAURADAS E COMPARAÇÕES
# ==========================================

VISUAL_DIR = OUTPUT / "visuals"
RESTORED_DIR = VISUAL_DIR / "restored"
COMPARISON_DIR = VISUAL_DIR / "comparisons"

methods_order = [
    "Gaussian",
    "Median",
    "Bilateral",
    "Adaptive",
    "Wavelet"
]

for folder in [RESTORED_DIR, COMPARISON_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

for method in methods_order:
    (RESTORED_DIR / method.lower()).mkdir(
        parents=True,
        exist_ok=True
    )


for _, row in validation_df.iterrows():

    noisy = cv2.imread(
        str(ROOT / row["noisy_path"]),
        cv2.IMREAD_COLOR
    )

    gt = cv2.imread(
        str(ROOT / row["gt_path"]),
        cv2.IMREAD_COLOR
    )

    outputs = {}

    for method in methods_order:

        params = best_configs[method]["params"]

        restored = apply_method(
            noisy,
            method,
            params,
            noise_variance
        )

        outputs[method] = restored

        cv2.imwrite(
            str(
                RESTORED_DIR
                / method.lower()
                / f"{row['folder']}.png"
            ),
            restored
        )

    # Converter imagens para RGB
    noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(
        1, 7,
        figsize=(22, 5)
    )

    axes[0].imshow(noisy_rgb)
    axes[0].set_title("Original ruidosa")

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
        COMPARISON_DIR / f"{row['folder']}.png",
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)


print("\n===== EXPERIMENTOS CONCLUÍDOS =====")
print("Métricas individuais:", OUTPUT / "validation_per_image.csv")
print("Resumo:", OUTPUT / "validation_summary.csv")
print("Configurações:", selection_path)
print("Comparações visuais:", COMPARISON_DIR)
print("Imagens restauradas:", RESTORED_DIR)