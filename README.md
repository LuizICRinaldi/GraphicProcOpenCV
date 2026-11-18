# GraphicProcOpenCV — Smartphone Image Denoising

Projeto acadêmico de Processamento de Imagens desenvolvido com Python, OpenCV e NumPy para restauração de fotografias com ruído real de aquisição.

O trabalho consiste no desenvolvimento de um sistema de processamento em lote capaz de reduzir o ruído presente em fotografias de smartphones, procurando preservar bordas, texturas e detalhes.

---

## 1. Identificação do projeto

- **Combinação:** 6
- **Problema:** Restauração de imagens com ruído real.
- **Subtema:** iPhone 7.
- **Código da câmera:** `IP`.
- **Dataset:** SIDD-Small — Smartphone Image Denoising Dataset.
- **Formato utilizado:** sRGB (PNG).

## 2. Objetivo

Desenvolver e comparar diferentes estratégias clássicas de redução de ruído em fotografias capturadas pelo iPhone 7.

O sistema recebe imagens ruidosas, aplica métodos de restauração e compara os resultados com imagens de referência (Ground Truth).

O projeto também contempla o desenvolvimento de um filtro adaptativo próprio, baseado em estatísticas locais de média e variância, conforme exigido pelo enunciado.

Não são utilizados ruídos artificiais, redes neurais, modelos pré-treinados ou serviços externos de inteligência artificial para realizar a restauração.

---

## 3. Dataset

Foi utilizada uma seleção do SIDD-Small contendo exclusivamente imagens capturadas pelo iPhone 7, identificadas pelo código `IP`.

### Caracterização dos dados

| Condição de iluminação | Pares |
|---|---:|
| Low Light (L) | 21 |
| Normal Brightness (N) | 33 |
| **Total** | **54** |

Cada par contém:

- `NOISY_SRGB`: fotografia original contendo ruído real de aquisição.
- `GT_SRGB`: fotografia de referência utilizada para avaliar a restauração.

Foram utilizadas somente as imagens no espaço de cor sRGB. Os arquivos Raw-RGB e metadados não fazem parte do processamento.

**Download do dataset:** [Google Drive — SIDD iPhone 7 sRGB]((https://drive.google.com/drive/folders/1vi2vPiz0L2r4eys9I4JpGYfF-9pWlnb-?usp=drive_link))

**Página oficial:** https://abdokamel.github.io/sidd/

O dataset não está armazenado diretamente neste repositório. Para executar o projeto, é necessário baixá-lo e extrair seu conteúdo conforme as instruções deste README.

### Estrutura esperada do dataset

```text
opencv/
└── SIDD_Small_sRGB_Only/
    └── data/
        ├── 0028_001_IP_00100_00160_5500_N/
        │   ├── GT_SRGB_010.PNG
        │   └── NOISY_SRGB_010.PNG
        ├── 0193_009_IP_00800_02000_3200_N/
        │   ├── GT_SRGB_010.PNG
        │   └── NOISY_SRGB_010.PNG
        └── ... (54 pastas)
```

Os nomes originais das pastas e arquivos devem ser preservados.

---

## 4. Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Linguagem principal |
| OpenCV | Leitura, filtragem e processamento de imagens |
| NumPy | Operações matriciais e estatísticas locais |
| Matplotlib | Visualização e comparação das imagens |
| Pandas | Organização de dados e tabelas experimentais |
| scikit-image | Cálculo das métricas de qualidade |
| scikit-learn | Separação dos conjuntos de dados |
| PyWavelets | Implementação de Wavelet Denoising |

---

## 5. Estrutura do projeto

```text
opencv/
│
├── src/
│   ├── inspect_dataset.py
│   ├── baseline.py
│   ├── split_dataset.py
│   ├── baseline_test.py
│   ├── gaussian_filter.py
│   ├── denoising_methods.py
│   ├── compare_filters.py
│   └── final_evaluation.py
│
├── results/
│   ├── pairs_ip.csv
│   ├── data_splits.csv
│   │
│   ├── baseline_per_image.csv
│   ├── baseline_summary.csv
│   ├── baseline_test_per_image.csv
│   ├── baseline_test_summary.csv
│   │
│   ├── gaussian/
│   │   ├── gaussian_per_image.csv
│   │   ├── gaussian_summary.csv
│   │   ├── restored/
│   │   └── comparisons/
│   │
│   ├── method_comparison/
│   │   ├── validation_per_image.csv
│   │   ├── validation_summary.csv
│   │   ├── validation_best_configs.json
│   │   └── visuals/
│   │
│   └── final_test/
│       ├── final_per_image.csv
│       ├── final_summary.csv
│       ├── restored/
│       └── comparisons/
│
├── SIDD_Small_sRGB_Only/   (dataset local)
├── .venv/                 (ambiente virtual local)
├── requirements.txt
├── .gitignore
└── README.md
```

As pastas `.venv/` e `SIDD_Small_sRGB_Only/` não são enviadas ao GitHub. O ambiente virtual é recriado localmente e o dataset é disponibilizado pelo Google Drive.

---

## 6. Instalação e configuração

### Pré-requisitos

- Python instalado.
- Git instalado.
- Acesso ao dataset SIDD-Small disponibilizado no Google Drive.

### 6.1. Clonar o repositório

```bash
git clone https://github.com/LuizICRinaldi/GraphicProcOpenCV.git opencv

cd opencv
```

### 6.2. Criar o ambiente virtual

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

### 6.3. Instalar as dependências

Com o ambiente virtual ativado:

```bash
pip install -r requirements.txt
```

### 6.4. Preparar o dataset

1. Acesse o link do Google Drive.
2. Baixe o arquivo ZIP contendo os pares do iPhone 7.
3. Extraia o conteúdo na raiz do projeto.
4. Verifique se a estrutura corresponde à seção 3 deste README.

Após a extração, o projeto deverá encontrar as 54 pastas automaticamente.

---

## 7. Metodologia e pipeline

O desenvolvimento foi dividido em etapas para garantir a organização dos dados, a implementação dos métodos e a avaliação experimental.

### 7.1. Inspeção e preparação do dataset

Arquivo: `src/inspect_dataset.py`

O programa:

- Localiza as pastas identificadas com `IP`.
- Identifica os arquivos `NOISY_SRGB` e `GT_SRGB`.
- Carrega as imagens utilizando OpenCV.
- Verifica dimensões e canais.
- Gera um catálogo dos pares válidos.

Saída:

```text
results/pairs_ip.csv
```

### 7.2. Separação dos dados

Arquivo: `src/split_dataset.py`

Os 54 pares foram divididos em três conjuntos:

| Conjunto | Pares | Cenas |
|---|---:|---:|
| Treinamento | 28 | 5 |
| Validação | 14 | 2 |
| Teste | 12 | 2 |
| **Total** | **54** | **9** |

A separação foi realizada agrupando os pares pelo identificador de cena, mantendo imagens associadas à mesma cena no mesmo conjunto.

Foram utilizadas sementes aleatórias fixas para permitir a reprodução da divisão.

Finalidade de cada conjunto:

- **Treinamento:** estimar parâmetros estatísticos do ruído.
- **Validação:** comparar métodos e selecionar configurações.
- **Teste:** realizar a avaliação final, sem utilizar previamente seus resultados para ajustar os parâmetros.

Saída:

```text
results/data_splits.csv
```

### 7.3. Estimativa da variância do ruído

Arquivo: `src/denoising_methods.py`

A variância do ruído é estimada utilizando os 28 pares do conjunto de treinamento.

Inicialmente, calcula-se a diferença entre a imagem ruidosa e sua referência:

D = I - G

Em seguida, calcula-se a variância dessas diferenças separadamente para os canais BGR:

Var(D) = E[D²] - E[D]²

Na execução utilizada para a avaliação final, foram obtidas as seguintes estimativas:

| Canal | Variância estimada |
|---|---:|
| B | 104,5734 |
| G | 50,6784 |
| R | 144,7307 |

Essa estimativa é utilizada pelo filtro adaptativo durante a restauração.

### 7.4. Baseline

Arquivos:
- `src/baseline.py`
- `src/baseline_test.py`

Antes da aplicação dos filtros, foram calculadas as métricas das imagens ruidosas em comparação com suas referências.

O baseline representa o desempenho inicial, sem restauração.

### 7.5. Métodos de redução de ruído

Arquivo: `src/denoising_methods.py`

Foram implementados e comparados cinco métodos clássicos:

**Filtro Gaussiano**

Realiza suavização espacial utilizando uma média ponderada dos pixels vizinhos.

Implementação auxiliar: `cv2.GaussianBlur()`.

**Filtro Mediano**

Substitui o valor de cada pixel pela mediana dos valores de sua vizinhança.

Implementação auxiliar: `cv2.medianBlur()`.

**Filtro Bilateral**

Considera a distância espacial e a semelhança de intensidade entre os pixels durante a suavização, buscando preservar descontinuidades e bordas.

Implementação auxiliar: `cv2.bilateralFilter()`.

**Filtro Adaptativo próprio**

Componente algorítmico desenvolvido pelo grupo.

Utiliza média e variância locais para decidir quanto suavizar cada pixel.

A implementação utiliza operações matriciais do NumPy e `cv2.boxFilter()` como operação auxiliar para calcular estatísticas da vizinhança.

O ganho adaptativo é calculado por:

g(p) = max(0, σ²L(p) - σ²n) / (σ²L(p) + ε)

O pixel restaurado é obtido por:

J(p) = μ(p) + g(p) × [I(p) - μ(p)]

Dessa maneira, a intensidade da suavização depende das características locais da imagem.

Foram investigadas janelas de 3×3, 5×5 e 7×7.

**Wavelet Denoising**

Alternativa adicional pesquisada pelo grupo.

Utiliza decomposição wavelet, thresholding dos coeficientes de detalhe e reconstrução da imagem.

Implementação utilizando a biblioteca PyWavelets.

---

## 8. Investigação experimental

Arquivo: `src/compare_filters.py`

A investigação experimental teve como objetivo comparar o desempenho dos cinco métodos de restauração e identificar quais configurações apresentavam melhores resultados.

Foram utilizadas as 14 imagens do conjunto de validação.

Foram investigadas 13 configurações:

| Método | Configurações |
|---|---|
| Gaussiano | 7×7, sigma 1.5 |
| Mediano | Kernels 3, 5 e 7 |
| Bilateral | Três combinações de parâmetros |
| Adaptativo | Janelas 3, 5 e 7 |
| Wavelet | Multipliers 0.6, 0.8 e 1.0 |

Cada configuração foi aplicada às 14 imagens, totalizando 182 avaliações de restauração.

### Métricas utilizadas

| Métrica | Objetivo | Melhor resultado |
|---|---|---|
| PSNR | Avaliar erro em relação à referência | Maior |
| SSIM | Avaliar similaridade estrutural | Maior |
| MAE | Avaliar diferença absoluta dos pixels | Menor |
| Edge MAE | Comparar mapas de magnitude dos gradientes | Menor |
| Tempo | Avaliar custo computacional | Menor |

O Edge MAE foi calculado comparando os gradientes das imagens restauradas e das referências, utilizando o operador Sobel.

Também foram geradas comparações visuais entre as imagens originais, restauradas e suas respectivas referências.

### Configurações selecionadas na validação

A configuração selecionada para cada método foi:

| Método | Configuração |
|---|---|
| Gaussian | Kernel 7×7, sigma 1.5 |
| Median | Kernel 7 |
| Bilateral | d=9, sigmaColor=75, sigmaSpace=75 |
| Adaptive | Janela 7×7 |
| Wavelet | db2, nível 2, multiplier 1.0 |

### Resultados da validação

| Método | PSNR | SSIM | MAE | Edge MAE |
|---|---:|---:|---:|---:|
| Gaussian | 35,3065 | 0,8557 | 3,2701 | 5,2681 |
| Median | 36,6139 | 0,8942 | 2,7537 | 4,7378 |
| Bilateral | 36,8304 | 0,9060 | 2,7018 | 4,4536 |
| Adaptive | 34,8864 | 0,8548 | 3,2488 | 6,7274 |
| Wavelet | 33,4662 | 0,7924 | 3,9713 | 7,1251 |

O filtro Bilateral apresentou os melhores resultados gerais na validação.

As configurações selecionadas foram registradas em:

```text
results/method_comparison/validation_best_configs.json
```

Essas configurações foram posteriormente utilizadas na avaliação final.

---

## 9. Avaliação final

Arquivo: `src/final_evaluation.py`

Após a seleção das configurações utilizando o conjunto de validação, os cinco métodos foram avaliados no conjunto de teste, composto por 12 pares.

O conjunto de teste foi utilizado exclusivamente para a avaliação final.

Além das métricas quantitativas, foram salvas as imagens restauradas e as comparações visuais.

### Resultados gerais do conjunto de teste

| Método | PSNR | SSIM | MAE | Edge MAE | Tempo médio (ms) |
|---|---:|---:|---:|---:|---:|
| Original | 28,7432 | 0,5944 | 7,2119 | 23,1807 | — |
| Gaussian | 34,0290 | 0,8490 | 3,8112 | 10,4230 | 6,09 |
| Median | 33,2856 | 0,7975 | 4,2496 | 14,2822 | 796,27 |
| Bilateral | 34,6385 | 0,8635 | 3,5817 | 11,9437 | 124,43 |
| Adaptive | 33,6483 | 0,8376 | 3,9011 | 9,9746 | 621,62 |
| Wavelet | 32,3226 | 0,7861 | 4,6317 | 11,5507 | 1634,33 |

Os tempos representam as médias observadas nas execuções realizadas no ambiente de desenvolvimento e podem variar conforme o hardware e as condições de execução.

### Resultados por condição de iluminação

**PSNR e SSIM**

| Método | PSNR L | PSNR N | SSIM L | SSIM N |
|---|---:|---:|---:|---:|
| Original | 28,6744 | 28,7924 | 0,5633 | 0,6165 |
| Gaussian | 34,2035 | 33,9043 | 0,8302 | 0,8624 |
| Median | 33,4460 | 33,1711 | 0,7696 | 0,8174 |
| Bilateral | 34,8377 | 34,4962 | 0,8453 | 0,8765 |
| Adaptive | 34,1567 | 33,2852 | 0,8338 | 0,8403 |
| Wavelet | 32,5567 | 32,1554 | 0,7675 | 0,7994 |

### Discussão dos resultados

O filtro Bilateral apresentou o melhor desempenho quantitativo geral, alcançando os maiores valores de PSNR e SSIM e o menor MAE.

O filtro Gaussiano destacou-se pela velocidade de execução, apresentando o menor tempo médio entre os métodos.

O filtro Adaptativo desenvolvido pelo grupo apresentou o menor Edge MAE, indicando maior correspondência entre os mapas de gradientes das imagens restauradas e das referências.

Embora não tenha apresentado o melhor desempenho geral em PSNR e SSIM, o filtro adaptativo demonstrou capacidade de redução de ruído e um resultado interessante em relação à preservação de estruturas.

O filtro Mediano apresentou resultados intermediários, enquanto o Wavelet obteve o menor desempenho quantitativo entre os métodos avaliados nas configurações selecionadas.

Os resultados indicam que o método mais adequado depende do critério considerado: qualidade geral, velocidade de execução ou correspondência de bordas.

---

## 10. Limitações do experimento

Algumas limitações devem ser consideradas na interpretação dos resultados:

- O conjunto de teste contém apenas 12 pares distribuídos em duas cenas distintas.
- A quantidade reduzida de cenas limita a generalização dos resultados.
- A variância do ruído é estimada globalmente por canal, embora o ruído real possa variar conforme a intensidade da iluminação e as condições de aquisição.
- Foram investigadas quantidades limitadas de configurações para cada método.
- Os tempos de processamento foram obtidos em execuções pontuais e dependem do hardware utilizado.
- O Edge MAE mede a correspondência entre magnitudes de gradientes e não representa isoladamente uma garantia de preservação visual de todos os detalhes.

---

## 11. Como executar o projeto

Todos os comandos devem ser executados a partir da raiz do projeto (`opencv`), com o ambiente virtual ativado e o dataset preparado.

### Inspecionar o dataset

```bash
python src/inspect_dataset.py
```

### Calcular o baseline geral

```bash
python src/baseline.py
```

### Separar os dados

```bash
python src/split_dataset.py
```

### Calcular o baseline do teste

```bash
python src/baseline_test.py
```

### Executar os experimentos de validação

```bash
python src/compare_filters.py
```

### Executar a avaliação final

```bash
python src/final_evaluation.py
```

A avaliação final utiliza as configurações previamente selecionadas na validação, registradas em `validation_best_configs.json`.

---

## 12. Conclusão

O projeto desenvolveu uma pipeline completa de restauração em lote para fotografias com ruído real capturadas pelo iPhone 7.

Foram implementados e comparados cinco métodos clássicos de redução de ruído, incluindo um filtro adaptativo próprio baseado em estatísticas locais.

Os resultados demonstraram que todos os métodos melhoraram as imagens em relação ao baseline original.

O Bilateral apresentou o melhor desempenho geral nas métricas PSNR, SSIM e MAE, enquanto o Gaussiano apresentou o menor tempo médio de processamento.

O filtro adaptativo desenvolvido pelo grupo apresentou o menor erro de gradientes entre os métodos, demonstrando uma característica relevante para a investigação da preservação de bordas e estruturas.

A investigação também demonstrou que a escolha do método depende do equilíbrio desejado entre redução de ruído, preservação de detalhes e custo computacional.

---

## 14. Autoria

**Projeto:** Processamento de Imagens.

**Repositório:** https://github.com/LuizICRinaldi/GraphicProcOpenCV

**Integrantes:**
Abraao, Luiz Inácio e Nicolas

**Instituição:** Unisinos

**Ano:** 2026.
