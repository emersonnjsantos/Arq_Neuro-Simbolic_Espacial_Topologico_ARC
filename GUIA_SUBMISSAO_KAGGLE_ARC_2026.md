# Guia Estratégico de Submissão — ARC Prize 2026 (Paper Track)

**Projeto:** Topological Neuro-Symbolic Engine (T-NSE)  
**Autor:** Emerson Noé José dos Santos  
**Repositório:** [Arq_Neuro-Simbolic_Espacial_Topologico_ARC](https://github.com/emersonnjsantos/Arq_Neuro-Simbolic_Espacial_Topologico_ARC)  
**Notebook do Projeto:** [`Neuro-Simbolic_Espacial_Topologico_ARC.ipynb`](./Neuro-Simbolic_Espacial_Topologico_ARC.ipynb)  
**Relatório Oficial (Inglês, <= 1.500 palavras):** [`SUBMISSION_ARC_PRIZE_2026_REPORT.md`](./SUBMISSION_ARC_PRIZE_2026_REPORT.md)  
**Imagem de Capa (Media Gallery):** [`cover_image_arc_prize_2026.jpg`](./cover_image_arc_prize_2026.jpg)  

---

## 1. Visão Geral da Competição & Premiação

* **Competição:** ARC Prize 2026 — Paper Track (Kaggle Hackathon)
* **Objetivo:** Documentar a abordagem conceitual para resolver o [ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) ou [ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3).
* **Premiação (Total: US$ 75.000):**
  * 🥇 **1º Lugar:** US$ 50.000
  * 🥈 **2º Lugar:** US$ 20.000
  * 🥉 **3º Lugar:** US$ 5.000
* **Data Limite Final:** 9 de Novembro de 2026 (23:59 UTC).

---

## 2. Alinhamento Estrito com os 6 Critérios de Avaliação (Nota 0 a 5)

| Critério | O que os Juízes Avaliam | Como a Nossa Abordagem (T-NSE) Responde |
|---|---|---|
| **1. Precisão (Accuracy)** | Desempenho no ranking e fidelidade aos pares de teste. | Validação empírica determinística: o motor de execução testa hipóteses contra os pares de treino ($Loss = 0$) antes de emitir a previsão para o teste oculto. |
| **2. Universalidade** | O método se aplica a outros problemas além de puzzles do ARC? | A representação por *Topological Scene Graphs (TSG)* e decomposição em objetos é a base para raciocínio relacional em robótica (manipulação espacial), CAD/automação e raciocínio físico de senso comum. |
| **3. Progresso** | Aumenta a probabilidade de alguém bater 85%+ no ARC? | Supera os dois maiores gargalos atuais: elimina as alucinações espaciais dos LLMs e poda drasticamente a explosão combinatória da síntese simbólica por força bruta através de invariantes topológicos. |
| **4. Teoria** | Explica **POR QUE** funciona (fundamentação matemática/cognitiva)? | Baseado nos *Core Knowledge Priors* (Spelke / Chollet): Coesão de objetos, persistência espacial, homomorfismos discretos e topologia de conectividade-4. |
| **5. Completude** | Detalha toda a arquitetura de ponta a ponta? | Pipeline completo: Percepção (OpenCV) $\rightarrow$ Grafo de Cena $\rightarrow$ Poda Heurística $\rightarrow$ DSL Simbólica $\rightarrow$ Verificação Determinística. |
| **6. Novidade** | Quão inovador é em relação ao estado da arte público? | Abordagem verdadeiramente neuro-simbólica centrada em invariantes topológicos discretos (Euler, conectividade, casca convexa, translação vetorial) em vez de simples LLMs textuais ou busca cega. |

---

## 3. Passo a Passo para Submeter no Kaggle

### Passo 1: Acessar a Página de Projetos do Kaggle
1. Acesse o link oficial: [https://www.kaggle.com/competitions/arc-prize-2026-paper-track/projects](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/projects).
2. Clique no botão azul **"New Report"** (Novo Relatório).

### Passo 2: Preencher o Título e Subtítulo
* **Título (Title):**  
  `Topological Neuro-Symbolic Engine (T-NSE): Object-Centric Spatial Induction for ARC-AGI`
* **Subtítulo (Subtitle):**  
  `A theoretically grounded neuro-symbolic framework decoupling discrete topological perception, relational scene graphs, and verified program synthesis for ARC-AGI-2 and ARC-AGI-3.`
* **Trilha (Track):** Selecione **Main Track**.

### Passo 3: Colar o Conteúdo do Relatório
* Abra o arquivo [`SUBMISSION_ARC_PRIZE_2026_REPORT.md`](./SUBMISSION_ARC_PRIZE_2026_REPORT.md).
* Copie o texto em Markdown e cole diretamente no editor do Kaggle Report.
* **Verificação de Palavras:** O relatório possui aproximadamente **1.270 palavras**, respeitando com folga a margem obrigatória de **1.500 palavras**!

### Passo 4: Fazer Upload da Imagem de Capa (Media Gallery)
* O Kaggle exige obrigatoriamente uma imagem de capa na seção **Media Gallery**.
* Já geramos e salvamos em alta resolução o arquivo:  
  `D:\NEURO_SIMBOLIC_ESPACILA_ARC\cover_image_arc_prize_2026.jpg`
* Arraste esse arquivo para o campo de Capa/Galeria no Kaggle.

### Passo 5: Vincular o Caderno Público (Public Notebook)
1. No Kaggle, vá em **Code** $\rightarrow$ **New Notebook** (ou importe o arquivo [`Neuro-Simbolic_Espacial_Topologico_ARC.ipynb`](./Neuro-Simbolic_Espacial_Topologico_ARC.ipynb)).
2. Certifique-se de que a visibilidade do Notebook seja **Public** (Pública).
3. No seu Kaggle Report, na seção **Project Links**, vincule o link do seu notebook público.
4. *(Opcional)* Inclua também o link do repositório GitHub:  
   `https://github.com/emersonnjsantos/Arq_Neuro-Simbolic_Espacial_Topologico_ARC`

### Passo 6: Publicar e Enviar
1. Clique em **Save** (Salvar).
2. No canto superior direito, clique em **Submit** (Enviar) para concluir a participação oficial no Prêmio ARC 2026!
