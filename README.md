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

**Download do dataset:** [Google Drive — SIDD iPhone 7 sRGB]([DATASET Editado](https://drive.google.com/drive/folders/1vi2vPiz0L2r4eys9I4JpGYfF-9pWlnb-?usp=sharing))

**Página oficial:** https://abdokamel.github.io/sidd/

O dataset não está incluído diretamente neste repositório. Após baixar e extrair o arquivo ZIP, coloque a pasta `SIDD_Small_sRGB_Only` na raiz do projeto.

Estrutura esperada:

```text
opencv/
├── SIDD_Small_sRGB_Only/
│   └── data/
│       ├── 0028_001_IP_00100_00160_5500_N/
│       │   ├── GT_SRGB_010.PNG
│       │   └── NOISY_SRGB_010.PNG
│       └── ... (54 pastas)
```

Os nomes originais das pastas e arquivos devem ser preservados.

## 4. Tecnologias utilizadas

- Python
- OpenCV
- NumPy
- Matplotlib
- Pandas
- scikit-image
- scikit-learn

## 5. Estrutura do projeto

```text
opencv/
│
├── src/
│   ├── inspect_dataset.py
│   ├── baseline.py
│   ├── split_dataset.py
│   ├── baseline_test.py
│   └── gaussian_filter.py
│
├── results/
│   ├── pairs_ip.csv
│   ├── data_splits.csv
│   ├── baseline_per_image.csv
│   ├── baseline_summary.csv
│   ├── baseline_test_per_image.csv
│   ├── baseline_test_summary.csv
│   └── gaussian/
│       ├── gaussian_per_image.csv
│       ├── gaussian_summary.csv
│       ├── restored/
│       └── comparisons/
│
├── SIDD_Small_sRGB_Only/ (dataset local, não enviado ao GitHub)
├── .venv/ (ambiente virtual local, não enviado ao GitHub)
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Instalação e configuração

### Pré-requisitos

- Python instalado.
- Git instalado.
- Dataset SIDD-Small preparado conforme a estrutura descrita anteriormente.

### Clonar o repositório

```bash
git clone https://github.com/LuizICRinaldi/GraphicProcOpenCV.git opencv

cd opencv
```

### Criar o ambiente virtual

Windows PowerShell:

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
python -m venv .venv

.venv\Scripts\activate.bat
```

Linux/macOS:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Instalar dependências

```bash
pip install -r requirements.txt
```

## 7. Execução do projeto

Todos os comandos devem ser executados a partir da raiz do projeto (`opencv`), com o ambiente virtual ativado.

### 7.1. Inspeção do dataset

```bash
python src/inspect_dataset.py
```

Responsabilidades:

- Localizar as pastas identificadas com `IP`.
- Encontrar os arquivos Noisy sRGB e Ground Truth.
- Validar a leitura das imagens.
- Verificar dimensões e canais.
- Gerar um catálogo dos pares.

Saída:

```text
results/pairs_ip.csv
```

### 7.2. Baseline geral

```bash
python src/baseline.py
```

Calcula as métricas das 54 imagens ruidosas originais, sem aplicação de filtros.

Métricas calculadas:

- PSNR.
- SSIM.
- MAE.

Saídas:

```text
results/baseline_per_image.csv
results/baseline_summary.csv
```

### 7.3. Divisão dos dados

```bash
python src/split_dataset.py
```

Divide os pares em três conjuntos, agrupando imagens pela identificação da cena para evitar vazamento de dados.

Divisão obtida:

| Conjunto | Pares | Cenas |
|---|---:|---:|
| Treinamento | 28 | 5 |
| Validação | 14 | 2 |
| Teste | 12 | 2 |
| **Total** | **54** | **9** |

A divisão é reproduzível por meio das sementes aleatórias registradas no código.

Saída:

```text
results/data_splits.csv
```

### 7.4. Baseline do conjunto de teste

```bash
python src/baseline_test.py
```

Calcula as métricas das 12 imagens originais do conjunto de teste.

Saídas:

```text
results/baseline_test_per_image.csv
results/baseline_test_summary.csv
```

### 7.5. Experimentos com filtro Gaussiano

```bash
python src/gaussian_filter.py
```

Testa diferentes combinações de tamanho de kernel e sigma utilizando as 14 imagens do conjunto de validação.

Configurações investigadas:

| Kernel | Sigma |
|---|---|
| 3 × 3 | 0.5, 1.0, 1.5 |
| 5 × 5 | 0.5, 1.0, 1.5 |
| 7 × 7 | 0.5, 1.0, 1.5 |

São calculadas métricas individuais, resultados agregados e tempo médio de processamento.

A configuração de melhor resultado inicial na validação foi:

- Kernel: 7 × 7.
- Sigma: 1.5.
- PSNR médio: 35.3065 dB.
- SSIM médio: 0.8557.
- MAE médio: 3.2701.
- Tempo médio: 7.5415 ms.

A configuração selecionada também é utilizada para gerar imagens restauradas e comparações visuais entre as imagens originais, restauradas e suas referências.

Saídas:

```text
results/gaussian/
├── gaussian_per_image.csv
├── gaussian_summary.csv
├── restored/
└── comparisons/
```

Os resultados de validação não representam a avaliação final no conjunto de teste.

## 8. Resultados iniciais do baseline

### Dataset completo

| Métrica | Média |
|---|---:|
| PSNR | 28.4965 dB |
| SSIM | 0.5228 |
| MAE | 7.3556 |

### Conjunto de teste

| Métrica | Média |
|---|---:|
| PSNR | 28.7432 dB |
| SSIM | 0.5944 |
| MAE | 7.2119 |

Esses valores representam as imagens originais ruidosas, antes da aplicação de qualquer filtro.

## 9. Métricas de avaliação

O projeto utiliza as seguintes métricas:

| Métrica | Objetivo | Melhor resultado |
|---|---|---|
| PSNR | Medir o erro em relação à referência | Maior |
| SSIM | Avaliar similaridade estrutural | Maior |
| MAE | Medir diferença absoluta entre pixels | Menor |
| Tempo | Avaliar custo computacional | Menor |

Também será investigada a preservação de bordas e detalhes por meio da comparação de gradientes.

As métricas serão calculadas individualmente por imagem e agregadas para comparação entre os métodos.

## 10. Métodos de restauração previstos

O projeto contempla a implementação e comparação das seguintes estratégias:

- Filtro Gaussiano.
- Filtro Mediano.
- Filtro Bilateral.
- Filtro adaptativo desenvolvido pelo grupo, baseado em estatísticas locais.
- Wavelet Denoising.

O filtro adaptativo é o componente algorítmico obrigatório e deverá ser desenvolvido e explicado pelo grupo.

## 11. Status do desenvolvimento

- [x] Preparação e inspeção do dataset.
- [x] Seleção dos 54 pares do iPhone 7.
- [x] Validação das imagens e geração do catálogo.
- [x] Cálculo do baseline geral.
- [x] Separação de treinamento, validação e teste.
- [x] Cálculo do baseline do conjunto de teste.
- [x] Implementação e experimentação inicial do filtro Gaussiano.
- [x] Comparação inicial de nove configurações Gaussianas.
- [x] Geração de imagens restauradas e comparações visuais do Gaussiano.
- [ ] Filtro Mediano.
- [ ] Filtro Bilateral.
- [ ] Filtro adaptativo próprio.
- [ ] Wavelet Denoising.
- [ ] Investigação experimental comparativa completa.
- [ ] Avaliação final no conjunto de teste.
- [ ] Relatório técnico.

## 12. Referências

ABDELHAMED, A.; LIN, S.; BROWN, M. S. A High-Quality Denoising Dataset for Smartphone Cameras. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

DOI: https://doi.org/10.1109/CVPR.2018.00182

Dataset SIDD: https://abdokamel.github.io/sidd/

## 13. Autoria

**Projeto acadêmico:** Processamento de Imagens.

**Integrantes:** preencher com os nomes dos integrantes.

**Instituição:** preencher com o nome da instituição.

**Ano:** 2026.
