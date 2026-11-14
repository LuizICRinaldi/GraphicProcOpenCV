from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ==========================================
# 1. CAMINHOS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "results" / "pairs_ip.csv"

OUTPUT_DIR = PROJECT_ROOT / "results"


# ==========================================
# 2. CARREGAR CATÁLOGO
# ==========================================

df = pd.read_csv(CSV_PATH)

print("Total de pares:", len(df))


# ==========================================
# 3. IDENTIFICAR CENAS
# ==========================================

# Separar os campos do nome da pasta
folder_parts = df["folder"].str.split("_")

# Segundo campo = identificador da cena
df["scene_id"] = folder_parts.apply(lambda parts: parts[1])

# Último campo = condição de iluminação
df["brightness"] = folder_parts.apply(lambda parts: parts[6])

print("Total de cenas:", df["scene_id"].nunique())


# ==========================================
# 4. SEPARAR TESTE (20%)
# ==========================================

test_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_val_indices, test_indices = next(
    test_splitter.split(df, groups=df["scene_id"])
)

train_val = df.iloc[train_val_indices].copy()
test = df.iloc[test_indices].copy()


# ==========================================
# 5. SEPARAR TREINAMENTO E VALIDAÇÃO
# ==========================================

validation_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=43
)

train_indices, validation_indices = next(
    validation_splitter.split(
        train_val,
        groups=train_val["scene_id"]
    )
)

train = train_val.iloc[train_indices].copy()
validation = train_val.iloc[validation_indices].copy()


# ==========================================
# 6. IDENTIFICAR OS CONJUNTOS
# ==========================================

train["split"] = "train"
validation["split"] = "validation"
test["split"] = "test"


# ==========================================
# 7. COMBINAR RESULTADOS
# ==========================================

split_df = pd.concat(
    [train, validation, test],
    ignore_index=True
)


# ==========================================
# 8. VERIFICAR VAZAMENTO DE CENAS
# ==========================================

train_scenes = set(train["scene_id"])
validation_scenes = set(validation["scene_id"])
test_scenes = set(test["scene_id"])

assert train_scenes.isdisjoint(validation_scenes)
assert train_scenes.isdisjoint(test_scenes)
assert validation_scenes.isdisjoint(test_scenes)

assert len(split_df) == len(df)

print("\nNenhuma cena compartilhada entre conjuntos.")


# ==========================================
# 9. SALVAR DIVISÃO
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / "data_splits.csv"

split_df.to_csv(output_path, index=False)


# ==========================================
# 10. APRESENTAR RESULTADOS
# ==========================================

print("\n===== DIVISÃO DO DATASET =====")

for name, group in split_df.groupby("split"):

    print(f"\n{name.upper()}")
    print("Pares:", len(group))
    print("Cenas:", group["scene_id"].nunique())

    print("\nIluminação:")
    print(group["brightness"].value_counts().to_string())

print("\nArquivo salvo:", output_path)