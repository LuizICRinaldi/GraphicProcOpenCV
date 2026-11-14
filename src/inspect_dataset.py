import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "SIDD_Small_sRGB_Only"
    / "Data"
)

print("Dataset encontrado:", DATA_DIR)

ip_folders = sorted([
    folder
    for folder in DATA_DIR.iterdir()
    if folder.is_dir()
    and folder.name.split("_")[2] == "IP"
])

print("Total de pastas IP:", len(ip_folders))

for folder in ip_folders:
    print(f"\nPasta: {folder.name}")

    for file in folder.iterdir():

        if file.suffix.lower() == ".png":
            print(f"  Arquivo: {file.name}")



# ==========================================
# 2. VALIDAR OS PARES DE IMAGENS
# ==========================================

pairs = []
errors = []

for folder in ip_folders:

    png_files = [
        file for file in folder.iterdir()
        if file.suffix.lower() == ".png"
    ]

    noisy_files = [
        file for file in png_files
        if "NOISY_SRGB" in file.stem.upper()
    ]

    gt_files = [
        file for file in png_files
        if "GT_SRGB" in file.stem.upper()
    ]

    # Verificar se encontrou exatamente um arquivo de cada tipo
    if len(noisy_files) != 1 or len(gt_files) != 1:
        errors.append(folder.name)
        print(f"Par inválido: {folder.name}")
        continue

    noisy_path = noisy_files[0]
    gt_path = gt_files[0]

    # Carregar imagens
    noisy = cv2.imread(
        str(noisy_path),
        cv2.IMREAD_UNCHANGED
    )

    ground_truth = cv2.imread(
        str(gt_path),
        cv2.IMREAD_UNCHANGED
    )

    # Verificar leitura
    if noisy is None or ground_truth is None:
        errors.append(folder.name)
        print(f"Erro de leitura: {folder.name}")
        continue

    # Verificar dimensões
    if noisy.shape != ground_truth.shape:
        errors.append(folder.name)
        print(f"Dimensões diferentes: {folder.name}")
        continue

    # Verificar canais
    if noisy.ndim != 3 or noisy.shape[2] != 3:
        errors.append(folder.name)
        print(f"Imagem inválida: {folder.name}")
        continue

    # Registrar par válido
    pairs.append({
        "folder": folder.name,
        "noisy_path": str(noisy_path.relative_to(PROJECT_ROOT)),
        "gt_path": str(gt_path.relative_to(PROJECT_ROOT)),
        "height": noisy.shape[0],
        "width": noisy.shape[1],
        "channels": noisy.shape[2],
        "dtype": str(noisy.dtype)
    })


# ==========================================
# 3. APRESENTAR RESULTADOS
# ==========================================

print("\n===== VALIDAÇÃO DO DATASET =====")

print("Pastas encontradas:", len(ip_folders))
print("Pares válidos:", len(pairs))
print("Erros encontrados:", len(errors))


# ==========================================
# 4. GERAR CATÁLOGO CSV
# ==========================================

output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(pairs)

df.to_csv(
    output_dir / "pairs_ip.csv",
    index=False
)

print("\nCatálogo pairs_ip.csv criado!")

# ==========================================
# 5. VISUALIZAR PRIMEIRO PAR DE IMAGENS
# ==========================================

if not df.empty:

    # Selecionar o primeiro par do catálogo
    sample = df.iloc[0]

    noisy_path = PROJECT_ROOT / sample["noisy_path"]
    gt_path = PROJECT_ROOT / sample["gt_path"]

    # Carregar imagens
    noisy = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)
    ground_truth = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

    # Converter BGR para RGB para visualização correta
    noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    ground_truth = cv2.cvtColor(
        ground_truth,
        cv2.COLOR_BGR2RGB
    )

    # Mostrar imagens lado a lado
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(noisy)
    plt.title("Noisy sRGB")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth)
    plt.title("Ground Truth sRGB")
    plt.axis("off")

    plt.suptitle(sample["folder"])
    plt.tight_layout()
    plt.show()

else:
    print("Nenhum par válido encontrado.")