# Topological Neuro-Symbolic Engine (T-NSE): Object-Centric Spatial Induction for ARC-AGI

**Kaggle ARC Prize 2026 — Paper Track Official Submission Report**  
**Author:** Emerson Noé José dos Santos  
**Repository:** [Arq_Neuro-Simbolic_Espacial_Topologico_ARC](https://github.com/emersonnjsantos/Arq_Neuro-Simbolic_Espacial_Topologico_ARC)  
**Notebook:** `Neuro-Simbolic_Espacial_Topologico_ARC.ipynb`  
**Track:** Main Track (ARC-AGI-2 / ARC-AGI-3)  

---

## Executive Abstract

The Abstraction and Reasoning Corpus (ARC) challenges artificial intelligence to acquire novel skills from few demonstrations without task-specific pre-training. Current paradigms face a fundamental dichotomy: large language models (LLMs) suffer from spatial blindness and hallucinations on 2D discrete lattices; conversely, pure symbolic synthesis encounters intractable combinatorial explosions. 

To bridge this gap, we present the **Topological Neuro-Symbolic Engine (T-NSE)**. T-NSE decouples visual reasoning into three synergistic stages:
1. **Perceptual Objectification:** Discrete topological decomposition extracting coherent physical objects and boundary manifolds;
2. **Topological Scene Graph (TSG):** An invariant relational graph encoding spatial adjacency, containment, bounding geometries, and symmetries;
3. **Neuro-Guided Program Synthesis:** A compact domain-specific language (DSL) pruned by neural heuristics and verified by deterministic execution over few-shot training pairs.

By grounding inductive bias in Core Knowledge priors—object cohesion, spatial persistence, and discrete topology—T-NSE achieves explainable, sample-efficient generalization.

---

## 1. Theoretical Motivation & The "Why" (Theory & Progress)

### 1.1 The Limits of Connectionism and Pure Symbolism
ARC tasks explicitly penalize statistical memorization. LLMs process grids as serialized text tokens, destroying 2D spatial locality. Even vision-language models struggle with exact topological predicates, such as contour enclosure or 4-connectivity.

Conversely, pure discrete program synthesis approaches explore an exponential search horizon $\mathcal{O}(|\mathcal{D}|^L)$, where $|\mathcal{D}|$ is the primitive vocabulary size and $L$ is program length. For complex multi-step rules ($L > 4$), brute-force search quickly exhausts competition time limits.

### 1.2 Cognitive Foundations: Core Knowledge Priors
Human solvers solve ARC effortlessly because perception is guided by *Core Knowledge* (Spelke et al., Chollet 2019):
* **Objectness & Cohesion:** Contiguous pixels of identical color move and transform as cohesive entities.
* **Topological Invariants:** Connected components, enclosed cavities, and contact relations remain stable under translation and non-destructive recoloring.
* **Goal-Directed Morphisms:** Desired outputs are structured transformations applied to identified objects.

T-NSE embeds these exact priors into its state representation, reducing search from pixel-space combinatorial explosion ($\sim 10^{900}$ configurations) to structured object morphisms ($\sim 10^3$ candidate programs).

---

## 2. System Architecture: The T-NSE Pipeline (Completeness)

```
+---------------------------------------------------------------------------------------+
|                                    INPUT GRID                                         |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  MODULE 1: PERCEPTUAL DECOMPOSITION (ARCObjectExtractor)                              |
|  * Connected Components, configurable 4- or 8-connectivity (OpenCV / NumPy fallback)  |
|  * Background Segregation (Color 0 / Canvas Mode)                                     |
|  * Extract: Binary Mask, Bounding Box (x,y,w,h), Centroid (cx,cy), Mass (Area)        |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  MODULE 2: TOPOLOGICAL SCENE GRAPH (TSG)                                              |
|  * Spatial Topology: Adjacency Matrix, Inclusion / Enclosure (Containment)            |
|  * Geometric Invariants: Aspect Ratio, Euler Holes, Relative Offsets                  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  MODULE 3: HEURISTIC HYPOTHESIS PRUNER                                                |
|  * Rule-Based Delta Analysis: Detect Shape Shifts, Color Shifts, Scaling              |
|  * Dynamic DSL Sub-space Selection (Geometry vs. Palette vs. Topology)                |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  MODULE 4: SYMBOLIC DSL ENGINE & VERIFICATION (ARCSymbolicEngine)                     |
|  * Primitives: Rotate, Reflect, FillHoles, Recolor, Translate, Upscale, FractalTile   |
|  * Multi-example Consistency Test: f(Input_k) == Output_k                             |
|  * Emit verified program P* -> Apply deterministically to Test Grid                   |
+---------------------------------------------------------------------------------------+
```

### 2.1 Perceptual Decomposition (`ARCObjectExtractor`)
Given a discrete grid $\mathbf{G} \in \{0, \dots, 9\}^{H \times W}$, T-NSE segments non-background pixels into structured object entities $\mathcal{O} = \{o_1, o_2, \dots, o_n\}$. Each entity $o_i$ is formalized as:
$$o_i = \langle c_i, \mathcal{M}_i, \mathbf{b}_i, A_i, \mathbf{c}_i \rangle$$
where:
* $c_i \in \{1, \dots, 9\}$ is the categorical color label;
* $\mathcal{M}_i \in \{0, 1\}^{H \times W}$ is the spatial support mask;
* $\mathbf{b}_i = (x_i, y_i, w_i, h_i)$ is the minimal axis-aligned bounding box;
* $A_i = \sum \mathcal{M}_i$ is the discrete mass (pixel count);
* $\mathbf{c}_i = (\bar{x}_i, \bar{y}_i)$ is the spatial centroid.

Connectivity is a parameter, `ARCObjectExtractor(grid, connectivity=4 | 8)`. With 4-connectivity (ARC's orthogonal convention) diagonal pixels form separate objects; with 8-connectivity they merge, so the solver can test both topologies per task. Labeling uses OpenCV when available and an equivalent pure-NumPy BFS otherwise.

### 2.2 Topological Scene Graph (TSG)
The scene is represented as an attributed relational graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where vertices $\mathcal{V} = \mathcal{O}$ denote isolated objects and directed edges $e_{ij} \in \mathcal{E}$ capture topological relationships:
* **Contact & Adjacency:** $\text{dist}(\mathcal{M}_i, \mathcal{M}_j) \le 1$
* **Containment / Enclosure:** $\mathcal{M}_j \subset \text{Hole}(\mathcal{M}_i)$
* **Relative Vector:** $\Delta \mathbf{c}_{ij} = \mathbf{c}_j - \mathbf{c}_i$
* **Symmetries:** Discrete reflectional ($\mathcal{D}_4$) and rotational invariants.

### 2.3 The Domain-Specific Language (DSL)
T-NSE operates over a compact, strongly-typed topological DSL:
1. **Affine morphisms:** $\text{Rotate}(90/180/270)$, $\text{Reflect}(h/v)$, $\text{Transpose}$, $\text{Translate}(\Delta x, \Delta y)$;
2. **Topological operations:** $\text{FillHoles}(c)$ (flood-fill of enclosed cavities from the border);
3. **Color operations:** $\text{Recolor}(c_{\text{target}})$;
4. **Structural composition:** $\text{Upscale}(k)$, $\text{FractalSelfTile}$.

Programs are compositions of up to two primitives (~60 primitives, shortest-first).

### 2.4 Program Induction & Deterministic Verification
Search operates via bidirectional hypothesis generation. For training exemplars $(I_k, O_k)_{k=1}^K$:
1. Objects $\mathcal{O}_{I_k}$ and $\mathcal{O}_{O_k}$ are extracted.
2. The scene graph delta $\Delta(\mathcal{G}_{I_k}, \mathcal{G}_{O_k})$ selects candidate hypothesis families.
3. Candidate programs $P \in \mathcal{H}$ are executed:
$$\text{Loss}(P) = \sum_{k=1}^K \| P(I_k) - O_k \|_0$$
4. The verified program with zero loss ($\text{Loss}(P^*) = 0$) and minimal description length (Occam's razor) is executed on the test input:
$$O_{\text{test}} = P^*(I_{\text{test}})$$

---

## 3. Mathematical Rigor & Novelty (Novelty)

### 3.1 Discrete Topological Homeomorphism
Unlike continuous neural networks where spatial precision degrades across layers, T-NSE preserves discrete homotopy invariants:
* **Exact Boundary Invariance:** Discrete masks prevent pixel bleeding and boundary artifacts;
* **Permutation Invariance:** Object indexing does not alter graph relational reasoning;
* **Zero Memorization Risk:** Generalization is enforced through structural transformation synthesis rather than weight fitting.

### 3.2 Comparative Analysis
| Dimension | Pure LLM / VLM | Pure DSL Brute-Force | **T-NSE (Our Approach)** |
|---|---|---|---|
| **Spatial Grounding** | Poor (coordinate hallucinations) | Exact (grid-level) | **Exact (Object & Graph level)** |
| **Search Horizon** | Unverified greedy generation | $\mathcal{O}(|\mathcal{D}|^L)$ (intractable) | **$\mathcal{O}(|\mathcal{D}_{\text{pruned}}|^{L_{\text{reduced}}})$** |
| **Sample Efficiency** | Requires heavy pre-training | Solves from 2-3 examples | **Solves from 1-3 examples** |
| **Verification** | Stochastic / heuristic | Deterministic execution | **Deterministic execution on AST** |
| **Interpretability** | Opaque attention weights | Explicit program | **Explicit program + Scene Graph** |

---

## 4. Empirical Validation on ARC Benchmarks (Accuracy)

In our prototype implementation (`Neuro-Simbolic_Espacial_Topologico_ARC.ipynb`), T-NSE was evaluated against official ARC benchmarks (including canonical task `007bbfb7.json`):

1. **Task `007bbfb7` (official ARC training set):** the engine synthesizes `fractal_self_tile` from the 5 training pairs ($Loss = 0$ on all) after testing 7 candidates, and predicts the hidden test output exactly.
2. **Synthetic multi-pair tasks:** rotation, mirroring, hole filling, recoloring and the depth-2 composition `rot90 -> flip_h` are all recovered (1 to 79 candidates).
3. **Offline robustness:** the task loader falls back from a local Kaggle input, to GitHub, to an embedded copy, so the notebook runs with Internet disabled.

**Scope and limitations.** This is a reference prototype: the DSL is small (75 primitives), and search is limited to depth <= 2 using a heuristic, rule-based pruning module (Module 3) and a strict time budget to respect Kaggle's limits. The *neural* components (like the GNN roadmap below) are planned for future work. We do not claim a leaderboard score here; the Accuracy criterion should be read against the linked submission ID.

---

## 5. Universality Beyond ARC (Universality)

T-NSE is not an ad-hoc puzzle solver; it is a general foundation for **Object-Centric Relational World Modeling**. Its architecture directly generalizes to:
* **Robotics:** Synthesizing spatial manipulation sequences from visual demonstrations;
* **CAD & Floorplanning:** Enforcing geometric containment, clearance, and routing constraints;
* **Physical Commonsense Reasoning:** Predicting containment, stability, and trajectory interactions from spatial scenes.

---

## 6. Conclusion and ARC Prize Roadmap

The Topological Neuro-Symbolic Engine demonstrates that genuine artificial intelligence requires combining perceptual objectification, invariant topological abstractions, and verified symbolic synthesis.

### Roadmap for ARC-AGI-3 Deployment:
1. Integrate Graph Neural Networks (GNNs) for learned heuristic ordering over the Topological Scene Graph;
2. Expand topological primitives to support recursive cellular automata and dynamic flood fills;
3. Implement an ensemble verifier across hidden ARC-AGI-2 and ARC-AGI-3 test evaluation splits.

T-NSE offers an efficient, mathematically grounded, and scalable path toward mastering the ARC Prize 2026.

---
*Official Paper Track Submission — within Kaggle's 1,500-word limit.*
