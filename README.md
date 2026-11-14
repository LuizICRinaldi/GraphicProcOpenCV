# GraphicProcOpenCV — Smartphone Image Denoising

Projeto acadêmico de Processamento de Imagens desenvolvido com Python, OpenCV e NumPy. O objetivo é desenvolver um sistema de restauração em lote para reduzir ruído real de aquisição em fotografias de smartphones, preservando bordas, texturas e detalhes.

## 1. Objetivo

Desenvolver e avaliar diferentes estratégias clássicas de redução de ruído em fotografias capturadas pelo iPhone 7.

O sistema recebe imagens ruidosas (Noisy sRGB), aplica técnicas de restauração e compara os resultados com imagens de referência (Ground Truth).

O projeto também contempla a implementação de um filtro adaptativo baseado em estatísticas locais, conforme exigido no trabalho.

## 2. Identificação do projeto

- **Combinação:** 6
- **Problema:** Restauração de imagens com ruído real.
- **Subtema:** iPhone 7.
- **Código da câmera:** `IP`.
- **Dataset:** SIDD-Small — Smartphone Image Denoising Dataset.
- **Formato utilizado:** sRGB (PNG).

## 3. Dataset

Foi utilizada uma seleção do SIDD-Small contendo exclusivamente imagens capturadas pelo iPhone 7 (`IP`).

O conjunto contém 54 pares de imagens:

| Condição de iluminação | Pares |
|---|---:|
| Low Light (L) | 21 |
| Normal Brightness (N) | 33 |
| **Total** | **54** |

Cada par contém:

- `NOISY_SRGB`: imagem original com ruído real de aquisição.
- `GT_SRGB`: imagem de referência (Ground Truth).

Não são adicionados ruídos artificialmente às imagens.

**Download do dataset:** [Google Drive — SIDD iPhone 7 sRGB](https://drive.google.com/drive/folders/1vi2vPiz0L2r4eys9I4JpGYfF-9pWlnb-?usp=sharing)

**Página oficial:** https://abdokamel.github.io/sidd/

O dataset não está incluído diretamente neste repositório. Após baixar e extrair o arquivo ZIP, coloque a pasta `SIDD_Small_sRGB_Only` na raiz do projeto.

Estrutura esperada:

```text
opencv/
├── SIDD_Small_sRGB_Only/
│   └── data/
│       ├── 0028_