# Topological Neuro-Symbolic Engine (T-NSE): Indução Espacial Centrada em Objetos para o Abstraction and Reasoning Corpus

**Artigo Técnico Oficial — ARC Prize 2026 (Paper Track)**  
**Trilha:** Main Track (ARC-AGI-2 / ARC-AGI-3)  
**Autor:** Emerson Noé José dos Santos  
**Comprovativo Oficial de Submissão (Main Track ID):** `57107077` (T-NSE v6)  
**Notebook do Projeto:** `Neuro-Simbolic_Espacial_Topologico_ARC.ipynb`  
**Repositório Oficial:** [github.com/emersonnjsantos/Arq_Neuro-Simbolic_Espacial_Topologico_ARC](https://github.com/emersonnjsantos/Arq_Neuro-Simbolic_Espacial_Topologico_ARC)  

---

## Resumo Executivo

O *Abstraction and Reasoning Corpus* (ARC) desafia os sistemas de inteligência artificial a adquirir novas habilidades a partir de poucas demonstrações, sem a possibilidade de memorização prévia via treinamento em larga escala. Os paradigmas dominantes enfrentam uma dicotomia fundamental: de um lado, os Grandes Modelos de Linguagem (LLMs) e Modelos de Visão-Linguagem (VLMs) sofrem de cegueira espacial e alucinações de coordenadas devido à serialização unidimensional de grades bidimensionais; de outro, a síntese de programas puramente simbólica por força bruta colapsa diante da explosão combinatória $\mathcal{O}(|\mathcal{D}|^L)$.

Para superar esse impasse, apresentamos o **Topological Neuro-Symbolic Engine (T-NSE)**. O T-NSE desacopla o raciocínio visual em três estágios complementares fundamentados nos princípios cognitivos de *Core Knowledge*:
1. **Percepção e Decomposição Topológica:** Decomposição da grade discreta em objetos físicos invariantes e contornos conexos através de conectividade configurável (4 ou 8 direções);
2. **Grafo de Cenas Topológico (TSG):** Construção de um grafo relacional que codifica adjacência espacial, contenção em cavidades internas (furos topológicos) e vetores de deslocamento relativo;
3. **Síntese de Programas com Poda Heurística e Verificação Determinística:** Busca gulosa de menor comprimento sobre uma Linguagem de Domínio Específico (DSL) fortemente tipada, podada por heurísticas relacionais e aceita estritamente se reproduzir 100% dos pares de treino com perda zero ($\text{Loss} = 0$).

A arquitetura foi implementada de ponta a ponta em Python/NumPy, validada em tarefas canônicas do ARC (incluindo a tarefa oficial `007bbfb7`) e executada de forma offline-safe no ambiente de avaliação do Kaggle, obtendo a submissão verificada sob o **Submission ID oficial `57107077`**.

---

## 1. Introdução e Motivação Teórica (Theory & Progress)

### 1.1 As Limitações do Conexionismo Puro e do Simbolismo Cego
O ARC (Chollet, 2019) foi concebido para medir a *eficiência na aquisição de habilidades* em vez de meras habilidades cristalizadas. Modelos conexionistas profundos e LLMs autorregressivos tratam grades $H \times W$ como sequências lineares de inteiros ou tokens. Essa linearização destrói a localidade espacial bidimensional fundamental, tornando os modelos incapazes de deduzir relações topológicas elementares como "estar contido dentro de uma cavidade", "tocar na borda externa" ou "conservar a simetria reflexiva".

Por outro lado, abordar o ARC exclusivamente através de busca exaustiva por força bruta em linguagens de domínio específico implica em um horizonte exponencial de busca:
$$\mathcal{N}(L) = \sum_{d=1}^{L} |\mathcal{D}|^d$$
Para um vocabulário modesto de $|\mathcal{D}| = 75$ primitivas, uma composição de profundidade $L = 3$ exige testar mais de $420.000$ candidatos por tarefa. Em competições como o ARC-AGI-2, onde mais de 100 tarefas devem ser avaliadas sob estritos limites de tempo computacional (sem acesso à Internet), a busca cega falha inevitavelmente por estouro de tempo (*timeout*).

### 1.2 Fundamentação Cognitiva: Priors de Core Knowledge
Solucionadores humanos resolvem a maioria dos puzzles do ARC em segundos sem necessidade de milhares de exemplos de treino. Esse desempenho decorre do ancoramento em *Priors de Conhecimento Central* (*Core Knowledge Priors*, Spelke & Kinzler, 2007; Chollet, 2019):
* **Coesão e Identidade de Objeto:** Pixels contíguos de mesma cor movem-se e comportam-se como entidades físicas sólidas e indivisíveis;
* **Invariantes Topológicos:** Conectividade, contorno fechado e número de cavidades internas (análogo à característica de Euler discreta) persistem sob operações de translação, rotação e recolorização;
* **Morfismos Orientados a Metas:** A transformação entre entrada e saída expressa operações estruturadas sobre entidades identificadas, e não mutações aleatórias pixel a pixel.

O T-NSE incorpora formalmente esses três axiomas em sua arquitetura, transpondo o problema de um espaço de busca combinatório de pixels ($\sim 10^{900}$ combinações possíveis em grades $30 \times 30$) para o espaço de morfismos de grafos de objetos ($\sim 10^3$ programas estruturados).

---

## 2. Formalização Matemática da Arquitetura (Completeness & Novelty)

O pipeline do T-NSE opera em um fluxo determinístico sequencial composto por três módulos integrados:

```
+-----------------------------------------------------------------------------------------+
|                                  GRADE DE ENTRADA G                                     |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|  MÓDULO 1: DECOMPOSIÇÃO TOPOLÓGICA (ARCObjectExtractor)                                 |
|  * Segmentação multicamada por cor (c ∈ {1..9}, fundo c=0)                              |
|  * Rotulagem de componentes conexos (4- ou 8-conectividade com fallback BFS NumPy)      |
|  * Formalização: o_i = ⟨ c_i, M_i, b_i, A_i, c_i ⟩                                      |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|  MÓDULO 2: GRAFO DE CENAS TOPOLÓGICO - TSG (build_scene_graph)                          |
|  * Detecção de furos por flood-fill de borda (enclosed_cells)                           |
|  * Relações tipadas: Adjacência (dist ≤ 1) e Contenção (M_j ⊂ Furo(M_i))                |
|  * Vetor relativo de centroide: Δc_ij = c_j - c_i                                       |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|  MÓDULO 3: PODA HEURÍSTICA DE HIPÓTESES (prune_dsl)                                     |
|  * Análise de deltas entre pares de treino (mudança de forma, escala, paleta)           |
|  * Redução seletiva do vocabulário da DSL (ex: de 75 para 3-6 primitivas ativas)        |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|  MÓDULO 4: MOTOR SIMBÓLICO E VERIFICAÇÃO DETERMINÍSTICA (ARCSymbolicEngine)             |
|  * Enumeração gulosa (menor comprimento primeiro, depth 1 -> depth 2)                   |
|  * Critério de aceitação: Loss = 0 em TODOS os pares de treino                          |
|  * Aplicação determinística à grade de teste oculta: O_test = P*(I_test)                |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|  MÓDULO 5: GERADOR DE SUBMISSÃO KAGGLE OFICIAL (submission.json)                        |
|  * Busca dinâmica por glob em /kaggle/input/                                            |
|  * Orçamento de tempo seguro por tarefa (max_time=1.5s, fallback garantido)            |
|  * Formato oficial ARC-AGI-2 com attempt_1 e attempt_2                                  |
+-----------------------------------------------------------------------------------------+
```

### 2.1 Percepção Discreta e Extração de Objetos
Dada uma grade bidimensional $\mathbf{G} \in \{0, \dots, 9\}^{H \times W}$, o sistema isola o plano de fundo ($c = 0$) e segmenta cada cor foreground $c \in \{1, \dots, 9\}$ em componentes conectados. Cada objeto $o_i$ é rigorosamente modelado pela 5-tupla:

$$o_i = \langle c_i, \mathcal{M}_i, \mathbf{b}_i, A_i, \mathbf{c}_i \rangle$$

Onde:
* $c_i \in \{1, \dots, 9\}$ é a etiqueta categórica da cor;
* $\mathcal{M}_i \in \{0, 1\}^{H \times W}$ é a máscara booleana de suporte espacial;
* $\mathbf{b}_i = (x_i, y_i, w_i, h_i)$ é a caixa delimitadora mínima (*bounding box*);
* $A_i = \sum_{x,y} \mathcal{M}_i(x,y)$ é a massa discreta (contagem de pixels);
* $\mathbf{c}_i = (\bar{x}_i, \bar{y}_i)$ é o centroide contínuo de massa.

**Invariância de Conectividade:** A topologia de adjacência pode ser operada sob 4-conectividade (padrão estrito ortogonal do ARC) ou 8-conectividade (incluindo diagonais), implementada através de `cv2.connectedComponents` com fallback automático para busca em largura pura (BFS) sobre NumPy, eliminando dependências rígidas de sistema operacional.

### 2.2 O Grafo de Cenas Topológico (TSG)
A percepção visual é estruturada no grafo atribuído direcionado $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, onde os vértices $\mathcal{V} = \{o_1, \dots, o_n\}$ representam os objetos identificados e as arestas direcionadas $e_{ij} = (o_i, o_j, r, \Delta \mathbf{c}_{ij}) \in \mathcal{E}$ definem predicados espaciais e topológicos estritos:

1. **Contenção / Enclosure (`contains`):** O objeto $o_j$ está inteiramente contido dentro de uma cavidade topológica fechada (furo) de $o_i$. O cálculo das cavidades é computado pelo operador `enclosed_cells(\mathcal{M}_i)`: realiza-se um flood-fill a partir das quatro bordas da grade sobre os pixels vazios; qualquer célula não-máscara inalcançável pelo exterior constitui um furo enclausurado. Cada nó armazena seu número de células de cavidade ($n_{\text{hole}}$).
2. **Adjacência Ortogonal (`adjacent`):** As máscaras de suporte de $o_i$ e $o_j$ tocam-se ortogonalmente com distância de Manhattan menor ou igual a 1 ($\text{dist}(\mathcal{M}_i, \mathcal{M}_j) \le 1$).
3. **Vetor de Posição Relativa (`delta`):** Diferença euclidiana entre os centroides: $\Delta \mathbf{c}_{ij} = \mathbf{c}_j - \mathbf{c}_i$.

### 2.3 Linguagem de Domínio Específico (DSL) Tipada
O T-NSE define uma DSL compacta e estritamente tipada contendo 75 primitivas fundamentais organizadas em cinco famílias funcionais:

| Família | Primitivas | Cardinalidade |
|---|---|:---:|
| **Afins (Grupo Diédrico $\mathcal{D}_4$)** | `rot90`, `rot180`, `rot270`, `flip_h`, `flip_v`, `transpose` | 6 |
| **Topológicas** | `fill_holes(c)`: preenche furos fechados com cor $c \in \{1..9\}$ | 9 |
| **Colorimétricas (Paleta)** | `recolor_all(c)`: substitui pixels foreground pela cor $c \in \{1..9\}$ | 9 |
| **Geométricas (Translação)** | `translate(dx, dy)` para $dx, dy \in \{-3..3\} \setminus \{(0,0)\}$ | 48 |
| **Estruturais / Fractais** | `fractal_self_tile` (auto-similaridade Kronecker), `upscale(k)` para $k \in \{2, 3\}$ | 3 |
| **Identidade** | `identity` | 1 |
| **Total** | | **75** |

Duas primitivas destacam-se pela elegância algébrica com produto de Kronecker:
```python
def fractal_self_tile(grid):
    g = np.array(grid, dtype=np.uint8)
    return np.kron((g != 0).astype(np.uint8), g)

def upscale(grid, k):
    return np.kron(np.array(grid, dtype=np.uint8), np.ones((k, k), dtype=np.uint8))
```

### 2.4 Poda Heurística de Hipóteses (TSG / Delta Pruning)
Para impedir a explosão combinatória, a função `prune_dsl(train_pairs)` inspeciona as propriedades dimensionais e topológicas dos pares de demonstração antes de iniciar a busca:
* Se as dimensões de entrada e saída coincidem em todos os exemplos, excluem-se operadores de redimensionamento e fractais, preservando rotações, espelhamentos, preenchimento de furos e translações;
* Se as dimensões da grade sofrem transposição ($H_{out} = W_{in}$ e $W_{out} = H_{in}$), ativam-se estritamente rotações de 90°/270° e transposição matricial;
* Se as dimensões escalam exatamente por fator $k \in \{2, 3\}$, a busca é restrita à família `upscale(k)`;
* Se $H_{out} = H_{in}^2$ e $W_{out} = W_{in}^2$, ativa-se imediatamente `fractal_self_tile`;
* Primitivas de preenchimento (`fill_holes`) e recolorização (`recolor_all`) são filtradas para incluir apenas cores efetivamente presentes no conjunto de saída ($c \in \text{Cores}(\mathbf{G}_{out})$).

Essa poda reduz o catálogo ativo de 75 para **3 a 15 primitivas**, viabilizando a busca completa de profundidade 1 em milissegundos e profundidade 2 em menos de um segundo por tarefa.

### 2.5 Função de Perda e Verificação Determinística
Dada uma sequência candidata de primitivas $P = (f_1, f_2, \dots, f_d)$, o programa é executado de forma determinística sobre todos os pares de demonstração $\{(I_k, O_k)\}_{k=1}^{K}$. A função de perda é definida por:

$$\text{Loss}(P) = \sum_{k=1}^{K} \big\lVert P(I_k) - O_k \big\rVert_0 \quad (\text{com } \text{Loss} = \infty \text{ se } \text{shape}(P(I_k)) \ne \text{shape}(O_k))$$

O programa ótimo $P^*$ é aceito **exclusivamente se $\text{Loss}(P^*) = 0$ para todos os pares de treino**. O programa verificado é então compilado e aplicado à grade de teste oculta:

$$\mathbf{O}_{\text{test}} = P^*(I_{\text{test}})$$

---

## 3. Validação Empírica e Benchmark Oficial (Accuracy & Completeness)

A arquitetura foi submetida a baterias rigorosas de testes sintéticos e benchmarks reais do ARC:

| Caso de Teste | Regra Oculta | Primitivas Testadas | Resultado T-NSE | Status |
|---|---|:---:|:---:|:---:|
| **Tarefa Oficial `007bbfb7`** | Auto-replicação fractal de Kronecker | 2 candidatos | `fractal_self_tile` | **Exato ($Loss = 0$)** |
| **Sintético: Rotação 90°** | Rotação horária pura | 2 candidatos | `rot90` | **Exato ($Loss = 0$)** |
| **Sintético: Espelhamento Horizontal** | Reflexão em eixo vertical | 18 candidatos | `flip_h` | **Exato ($Loss = 0$)** |
| **Sintético: Preenchimento de Furo** | Cavidade oclusa preenchida com cor 3 | 6 candidatos | `fill_holes(3)` | **Exato ($Loss = 0$)** |
| **Sintético: Recolorização** | Mapeamento global para cor 5 | 55 candidatos | `recolor_all(5)` | **Exato ($Loss = 0$)** |
| **Sintético: Composição Afim** | Rotação 180° seguida de transposição | 9 candidatos | `rot180 -> transpose` | **Exato (Profundidade 2)** |

### Robustez Offline no Kaggle ARC-AGI-2
O Módulo 5 integra um gerador de submissão resiliente desenhado para cumprir integralmente todas as exigências técnicas da competição:
1. **Descoberta Dinâmica de Arquivos via `glob`:** Elimina caminhos rígidos locais, localizando automaticamente `arc-agi_test_challenges.json` em qualquer ponto da árvore `/kaggle/input/`;
2. **Orçamento de Tempo Controlado:** Limite de 1.5 segundos por tarefa com poda heurística ativa, processando mais de 100 tarefas em menos de 2 minutos sem risco de *timeout*;
3. **Fallback Garantido e Conformidade Estrita:** Cada entrada de teste gera rigorosamente `attempt_1` e `attempt_2` no formato JSON exigido, prevenindo desclassificações por tarefas faltantes.

---

## 4. Comprovativo de Elegibilidade Oficial (Submission ID)

Em estrito cumprimento das diretrizes de elegibilidade da Trilha de Artigos (*Paper Track Rule 2.1.b*), o código foi submetido e avaliado no ambiente oficial do Kaggle:

* **Competição Vinculada:** ARC Prize 2026 — ARC-AGI-2
* **Nome da Submissão:** `T-NSE v6`
* **Submission ID Oficial e Definitivo:** **`57107077`**
* **Status da Avaliação:** Concluída com sucesso no leaderboard oficial (*Scored / Complete*).

---

## 5. Universalidade Além do ARC (Universality)

O T-NSE não é apenas um solucionador ad-hoc de quebra-cabeças bidimensionais; ele constitui um arcabouço genérico para **Modelagem Relacional de Mundo Centrada em Objetos** (*Object-Centric World Modeling*):

1. **Robótica Autônoma e Manipulação Espacial:** Extração de primitivas de movimentação de braços robóticos a partir de demonstrações visuais, convertendo dados perceptuais em grafos de contato e trajetórias vetoriais sem retreinamento neural;
2. **Automação Industrial e Design Assistido (CAD):** Verificação geométrica de peças, roteamento de componentes em placas de circuito e garantia de restrições de folga e contenção;
3. **Raciocínio de Senso Comum Físico:** Predição de suporte mecânico, estabilidade de pilhas de blocos e oclusão de cavidades a partir de invariantes topológicos estáticos.

---

## 6. Escopo Honesto e Limitações (Limitations)

Como contribuição científica rigorosa, explicitamos as fronteiras do estado atual do protótipo:
* **Dimensão da DSL:** O catálogo de 75 primitivas cobre transformações geométricas globais e preenchimento de cavidades, mas tarefas que demandam simulação de física de fluidos deformáveis ou raciocínio aritmético avançado exigem operadores especializados;
* **Profundidade de Composição:** A busca determinística opera prioritariamente em profundidades $L \le 2$ para respeitar a cota de tempo da competição;
* **Roadmap de Expansão:** As próximas etapas incluem a incorporação de Redes Neurais em Grafos (GNNs) para guiar probabilisticamente a ordem de expansão dos ramos do TSG e a introdução de primitivas orientadas a objetos individuais (`crop_bbox`, `isolate_object` e operadores de gravidade direcional).

---

## 7. Conclusão

O desenvolvimento de sistemas dotados de verdadeira inteligência fluida exige abandonar a suposição de que tudo pode ser resolvido por interpolação estatística de tokens. O **Topological Neuro-Symbolic Engine (T-NSE)** demonstra que acoplar algoritmos de visão estruturada (que extraem entidades e topologias físicas) a um motor de busca simbólica determinístico com verificação estrita produz uma solução transparente, matematicamente sólida e sample-efficient para o ARC-AGI.

Ao ensinar o sistema a "enxergar" os objetos e suas relações topológicas antes de deduzir as regras de transformação, construímos uma ponte concreta entre o conexionismo perceptivo e a precisão do raciocínio simbólico.

---

### Referências Bibliográficas

1. **Chollet, F.** (2019). *On the Measure of Intelligence*. arXiv preprint arXiv:1911.01547.
2. **Spelke, E. S., & Kinzler, K. D.** (2007). *Core knowledge*. Developmental Science, 10(1), 89-96.
3. **ARC Prize Foundation.** (2024–2026). *ARC-AGI-2 & ARC-AGI-3 Competitions and Technical Documentation*. arcprize.org.
4. **Santos, E. N. J.** (2026). *Topological Neuro-Symbolic Engine (T-NSE): Object-Centric Spatial Induction for ARC-AGI*. Repositório oficial do GitHub e Kaggle Paper Track.
