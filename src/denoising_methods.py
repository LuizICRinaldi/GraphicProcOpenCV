from pathlib import Path

import cv2
import numpy as np
import pywt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ==========================================
# 1. ESTIMAR VARIÂNCIA DO RUÍDO
# ==========================================

def estimate_noise_variance(train_df, project_root):
    """
    Estima a variância do ruído por canal BGR,
    utilizando exclusivamente os pares de treinamento.
    """

    noise_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for _, row in train_df.iterrows():

        noisy_path = project_root / row["noisy_path"]
        gt_path = project_root / row["gt_path"]

        noisy = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

        if noisy is None or gt is None:
            raise ValueError(f"Erro ao ler: {row['folder']}")

        if noisy.shape != gt.shape:
            raise ValueError(f"Dimensões diferentes: {row['folder']}")

        difference = noisy.astype(np.float32) - gt.astype(np.float32)

        noise_sum += np.sum(difference ** 2, axis=(0, 1))
        total_pixels += noisy.shape[0] * noisy.shape[1]

    if total_pixels == 0:
        raise ValueError("Não existem pixels para estimar o ruído.")

    variance = noise_sum / total_pixels

    return variance


# ==========================================
# 2. FILTRO ADAPTATIVO
# ==========================================

def adaptive_filter(image, window, noise_variance):
    """
    Filtro adaptativo baseado em média e variância locais.

    A implementação é feita pelo grupo utilizando
    operações matriciais e boxFilter como auxiliar.
    """

    restored_channels = []

    for channel in range(3):

        source = image[:, :, channel].astype(np.float32)

        local_mean = cv2.boxFilter(
            source,
            cv2.CV_32F,
            (window, window),
            normalize=True
        )

        local_mean_squared = cv2.boxFilter(
            source ** 2,
            cv2.CV_32F,
            (window, window),
            normalize=True
        )

        local_variance = np.maximum(
            local_mean_squared - local_mean ** 2,
            0
        )

        estimated_variance = noise_variance[channel]

        gain = np.maximum(
            local_variance - estimated_variance,
            0
        ) / (local_variance + 1e-8)

        restored = (
            local_mean
            + gain * (source - local_mean)
        )

        restored_channels.append(restored)

    result = cv2.merge(restored_channels)

    return np.clip(result, 0, 255).astype(np.uint8)


# ==========================================
# 3. WAVELET DENOISING
# ==========================================

def wavelet_filter(image, multiplier, wavelet="db2", level=2):
    """
    Denoising por decomposição wavelet e thresholding suave.
    A intensidade do threshold é controlada pelo multiplier.
    """

    restored_channels = []

    for channel in range(3):

        source = image[:, :, channel].astype(np.float32)

        coefficients = pywt.wavedec2(
            source,
            wavelet=wavelet,
            level=level,
            mode="symmetric"
        )

        # Estimativa robusta do desvio-padrão do ruído
        finest_details = coefficients[-1]
        diagonal_details = finest_details[2]

        sigma = (
            np.median(np.abs(diagonal_details)) / 0.6745
        )

        threshold = (
            multiplier
            * sigma
            * np.sqrt(2 * np.log(source.size))
        )

        filtered_coefficients = [coefficients[0]]

        for detail_level in coefficients[1:]:

            thresholded = tuple(
                pywt.threshold(
                    detail,
                    threshold,
                    mode="soft"
                )
                for detail in detail_level
            )

            filtered_coefficients.append(thresholded)

        reconstructed = pywt.waverec2(
            filtered_coefficients,
            wavelet=wavelet,
            mode="symmetric"
        )

        reconstructed = reconstructed[
            :source.shape[0],
            :source.shape[1]
        ]

        restored_channels.append(reconstructed)

    result = cv2.merge(restored_channels)

    return np.clip(result, 0, 255).astype(np.uint8)


# ==========================================
# 4. EXECUTAR UM MÉTODO
# ==========================================

def apply_method(image, method, params, noise_variance=None):

    if method == "Gaussian":

        return cv2.GaussianBlur(
            image,
            (params["kernel"], params["kernel"]),
            sigmaX=params["sigma"]
        )

    elif method == "Median":

        return cv2.medianBlur(
            image,
            params["ksize"]
        )

    elif method == "Bilateral":

        return cv2.bilateralFilter(
            image,
            params["d"],
            params["sigmaColor"],
            params["sigmaSpace"]
        )

    elif method == "Adaptive":

        if noise_variance is None:
            raise ValueError(
                "O filtro adaptativo precisa da variância do ruído."
            )

        return adaptive_filter(
            image,
            params["window"],
            noise_variance
        )

    elif method == "Wavelet":

        return wavelet_filter(
            image,
            params["multiplier"],
            params["wavelet"],
            params["level"]
        )

    else:
        raise ValueError(f"Método desconhecido: {method}")


# ==========================================
# 5. MÉTRICAS
# ==========================================

def calculate_metrics(restored, ground_truth):

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

    # Comparação dos gradientes para investigar
    # a preservação de bordas e detalhes.
    restored_gray = cv2.cvtColor(
        restored, cv2.COLOR_BGR2GRAY
    )

    gt_gray = cv2.cvtColor(
        ground_truth, cv2.COLOR_BGR2GRAY
    )

    restored_x = cv2.Sobel(
        restored_gray, cv2.CV_32F, 1, 0, ksize=3
    )

    restored_y = cv2.Sobel(
        restored_gray, cv2.CV_32F, 0, 1, ksize=3
    )

    gt_x = cv2.Sobel(
        gt_gray, cv2.CV_32F, 1, 0, ksize=3
    )

    gt_y = cv2.Sobel(
        gt_gray, cv2.CV_32F, 0, 1, ksize=3
    )

    restored_gradient = cv2.magnitude(
        restored_x, restored_y
    )

    gt_gradient = cv2.magnitude(
        gt_x, gt_y
    )

    edge_mae = np.mean(
        np.abs(restored_gradient - gt_gradient)
    )

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "MAE": mae,
        "Edge_MAE": edge_mae
    }