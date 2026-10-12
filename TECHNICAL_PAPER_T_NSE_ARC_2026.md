# Topological Neuro-Symbolic Engine (T-NSE): Object-Centric Spatial Induction for the Abstraction and Reasoning Corpus

**Official Technical Paper — Kaggle ARC Prize 2026 (Paper Track)**  
**Track:** Main Track (ARC-AGI-2 / ARC-AGI-3)  
**Author:** Emerson Noé José dos Santos  
**Official Verified Submission ID (Main Track):** `57107077` (T-NSE v6)  
**Project Notebook:** `Neuro-Simbolic_Espacial_Topologico_ARC.ipynb`  
**Official Repository:** [github.com/emersonnjsantos/Arq_Neuro-Simbolic_Espacial_Topologico_ARC](https://github.com/emersonnjsantos/Arq_Neuro-Simbolic_Espacial_Topologico_ARC)  

---

## Executive Abstract

The *Abstraction and Reasoning Corpus* (ARC) challenges artificial intelligence systems to acquire novel skills from a handful of demonstrations without task-specific pre-training. Dominant AI paradigms face a fundamental dichotomy: large language models (LLMs) and vision-language models (VLMs) suffer from spatial blindness and coordinate hallucinations caused by serializing 2D discrete grids into 1D token sequences; conversely, pure brute-force symbolic program synthesis quickly collapses due to combinatorial explosion $\mathcal{O}(|\mathcal{D}|^L)$.

To bridge this divide, we introduce the **Topological Neuro-Symbolic Engine (T-NSE)**. T-NSE decouples visual reasoning into three synergistic stages grounded in cognitive *Core Knowledge priors*:
1. **Topological Object Perception:** Discrete decomposition of the input lattice into invariant physical entities and coherent boundary manifolds using configurable connectivity (4- or 8-connectivity);
2. **Topological Scene Graph (TSG):** Construction of an attributed relational graph capturing spatial adjacency, enclosure within interior cavities (topological holes), and relative centroid offset vectors;
3. **Hypothesis-Pruned Symbolic Synthesis with Deterministic Verification:** Shortest-first search over a compact, strongly-typed Domain-Specific Language (DSL), pruned by relational heuristics and strictly verified under zero loss ($\text{Loss} = 0$) across 100% of demonstration pairs.

The architecture was implemented end-to-end in Python/NumPy, validated on canonical ARC benchmarks (including official challenge `007bbfb7`), and executed offline inside Kaggle's evaluation environment, achieving an officially scored entry under **Submission ID `57107077`**.

---

## 1. Theoretical Motivation & The "Why" (Theory & Progress)

### 1.1 The Pitfalls of Connectionism and Blind Symbolism
ARC (Chollet, 2019) was designed to measure *skill-acquisition efficiency* rather than crystallized statistical knowledge. Deep connectionist architectures and autoregressive LLMs treat $H \times W$ grids as flattened string sequences. This linearization destroys fundamental two-dimensional topological locality, rendering models incapable of systematically verifying spatial predicates such as "lies inside a cavity", "touches the boundary", or "preserves bilateral symmetry".

On the other hand, addressing ARC through unguided brute-force search over a domain-specific language leads to an intractable combinatorial search space. The total number of candidate programs $N(L)$ evaluated up to compositional depth $L$ is given by the geometric progression of candidate compositions:

$$N(L) = |D|^1 + |D|^2 + \dots + |D|^L$$

Where $|D|$ represents the vocabulary of available DSL primitives. For a modest vocabulary of $|D| = 75$ primitives, searching compositions up to depth $L = 3$ requires testing no less than **427,575 candidate programs** per task (75 single-step + 5,625 two-step + 421,875 three-step programs). In competitions such as ARC-AGI-2, where over 100 hidden tasks must be solved within strict offline CPU quotas, blind search inevitably triggers runtime timeouts and out-of-memory errors.

### 1.2 Cognitive Foundations: Core Knowledge Priors
Human solvers solve most ARC tasks within seconds without thousands of gradient updates. This efficiency originates from *Core Knowledge Priors* (Spelke & Kinzler, 2007; Chollet, 2019):
* **Objectness & Cohesion:** Contiguous same-colored pixels move and transform as solid, indivisible physical objects;
* **Topological Invariants:** Connectivity, closed contours, and the number of enclosed cavities (discrete analogues of the Euler characteristic) persist across translation, rotation, and non-destructive recoloring;
* **Goal-Directed Morphisms:** Transformations between input and output represent structured operations applied to identified entities rather than arbitrary pixel-level noise.

T-NSE explicitly bakes these three axioms into its representational state, shrinking the search space from raw pixel lattice configurations (an astronomical scale of $\sim 10^{900}$ possible states on $30 \times 30$ grids) down to structured object graph morphisms (pruned down to $\sim 1,000$ verified candidate programs).

---

## 2. Mathematical Formalization of the Architecture (Completeness & Novelty)

T-NSE organizes spatial induction into an end-to-end deterministic pipeline structured across five sequential modules:

| Pipeline Stage | Implementation Module | Formal Operation & Mathematical Mechanism | Output Representation |
|---|---|---|---|
| **0. Raw Ingestion** | Input Grid $G$ | Parse raw task JSON into discrete 2D integer lattice (values 0–9) of dimension $H \times W$ | Raw Discrete Lattice |
| **Stage 1 · Perception** | `ARCObjectExtractor` | Multilayer color filtering; 4- or 8-connectivity component labeling (OpenCV / pure-NumPy BFS) | Entity Set $\mathcal{O} = \{o_1, \dots, o_n\}$ with physical attributes |
| **Stage 2 · Topology** | `build_scene_graph` | Boundary flood-fill cavity detection (`enclosed_cells`); typed relation extraction | Attributed Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with enclosure, adjacency, and offsets |
| **Stage 3 · Pruning** | `prune_dsl` | Pairwise training deltas (shape preservation, scale factor, transpositions, active colors) | Filtered Sub-DSL $\mathcal{D}^*$ (reduced from 75 down to 3–15 active primitives) |
| **Stage 4 · Synthesis** | `ARCSymbolicEngine` | Shortest-first enumerative search (depth 1–2); deterministic verification loss with 100% accuracy | Verified Deterministic Program $P^*$ |
| **Stage 5 · Inference** | Submission Generator | Apply $P^*(I_{\text{test}})$; compile `attempt_1` & `attempt_2` into `/kaggle/working/submission.json` | Scored Kaggle ARC-AGI-2 Submission |

### 2.1 Discrete Perception and Object Extraction
Given an input discrete lattice $G$ of dimensions $H \times W$, background cells ($c = 0$) are segregated, and each foreground color $c \in \{1, \dots, 9\}$ is segmented into connected components. Every detected object $o_i$ is formalized as a 5-tuple:

$$o_i = \langle c_i, M_i, b_i, A_i, c_i^* \rangle$$

Where each component describes a concrete physical property:
* **Color ($c_i$):** Categorical color label (integer $1 \le c_i \le 9$);
* **Spatial Mask ($M_i$):** Binary support matrix in $\{0, 1\}^{H \times W}$ denoting the exact pixel occupancy;
* **Bounding Box ($b_i$):** Minimal bounding box $(x_i, y_i, w_i, h_i)$;
* **Area or Mass ($A_i$):** Discrete pixel mass (total count of foreground cells);
* **Centroid ($c_i^*$):** Continuous spatial center of mass $(\bar{x}_i, \bar{y}_i)$.

**Configurable Discrete Topology:** Component connectivity is parameterized via `connectivity=4` (ARC's strict orthogonal convention) or `connectivity=8` (diagonal merging). The labeling module wraps `cv2.connectedComponents` when OpenCV is present, with an automatic fallback to pure NumPy Breadth-First Search (BFS) to guarantee zero-dependency reproducibility.

### 2.2 The Topological Scene Graph (TSG)
Visual perception is represented as a directed attributed relational graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where vertices $\mathcal{V} = \{o_1, \dots, o_n\}$ denote isolated physical objects and directed edges $e_{ij} = (o_i, o_j, r, \Delta \mathbf{c}_{ij}) \in \mathcal{E}$ encode exact spatial predicates:

1. **Topological Containment (`contains`):** Object $o_j$ is entirely enveloped within an interior cavity (hole) of $o_i$. Cavities are computed via `enclosed_cells(\mathcal{M}_i)`: a flood-fill from all four grid boundaries marks empty exterior pixels; any non-mask cell unreachable from the boundary constitutes an interior hole. Each node records its hole cell count ($n_{\text{hole}}$).
2. **Orthogonal Adjacency (`adjacent`):** Support masks of $o_i$ and $o_j$ touch with a Manhattan distance $\le 1$ ($\text{dist}(\mathcal{M}_i, \mathcal{M}_j) \le 1$).
3. **Relative Centroid Vector (`delta`):** The relative offset between centroids: $\Delta \mathbf{c}_{ij} = \mathbf{c}_j - \mathbf{c}_i$.

### 2.3 Strongly-Typed Domain-Specific Language (DSL)
T-NSE defines a compact, typed DSL comprising 75 primitives organized into five functional families:

| Family | Primitives | Cardinality |
|---|---|:---:|
| **Affine (Dihedral Group $\mathcal{D}_4$)** | `rot90`, `rot180`, `rot270`, `flip_h`, `flip_v`, `transpose` | 6 |
| **Topological** | `fill_holes(c)`: flood-fills enclosed cavities with color $c \in \{1..9\}$ | 9 |
| **Palette / Colorimetric** | `recolor_all(c)`: recolors all foreground cells to color $c \in \{1..9\}$ | 9 |
| **Geometric (Translation)** | `translate(dx, dy)` for $dx, dy \in \{-3..3\} \setminus \{(0,0)\}$ | 48 |
| **Structural / Fractal** | `fractal_self_tile` (Kronecker self-tiling), `upscale(k)` for $k \in \{2, 3\}$ | 3 |
| **Identity** | `identity` | 1 |
| **Total** | | **75** |

Two structural primitives leverage matrix algebra via Kronecker products ($\otimes$), capturing fractal self-similarity and discrete resolution scaling:

| Structural Operator | Transformation Description | Exact Matrix Formulation |
|---|---|---|
| `fractal_self_tile(G)` | Replaces each active foreground cell with a full replica of grid $G$, creating fractal self-similarity | $G_{\text{out}} = (G > 0) \otimes G$ <br>*(Kronecker product of the binary active mask by the original grid)* |
| `upscale(G, k)` | Uniform nearest-neighbor magnification by integer scale factor $k \in \{2, 3\}$ | $G_{\text{out}} = G \otimes 1_{k \times k}$ <br>*(Kronecker product of the grid by a unit block of size $k \times k$)* |

### 2.4 Heuristic Hypothesis Pruning (TSG / Delta Pruning)
To prevent combinatorial explosion, `prune_dsl(train_pairs)` inspects geometric differences between input and output demonstration pairs before program synthesis begins:
* **Shape Preservation:** If input and output shapes match across all pairs, scaling and fractal operators are discarded, retaining rotations, reflections, cavity filling, and translations;
* **Transposition Detection:** If dimensions transpose ($H_{out} = W_{in}$ and $W_{out} = H_{in}$), the active set is restricted to 90°/270° rotations and matrix transposition;
* **Scaling Detection:** If dimensions scale by an integer factor $k \in \{2, 3\}$, search is narrowed strictly to `upscale(k)`;
* **Fractal Expansion:** If $H_{out} = H_{in}^2$ and $W_{out} = W_{in}^2$, the engine activates `fractal_self_tile`;
* **Palette Pruning:** `fill_holes(c)` and `recolor_all(c)` are pruned to only test colors that actually appear in the target demonstration grids ($c \in \text{Colors}(G_{out})$).

This pruning heuristic reduces the active primitive catalog from 75 down to **3 to 15 candidates**, enabling complete depth-1 search in milliseconds and depth-2 synthesis in under one second.

### 2.5 Deterministic Verification & Loss Function
Given a candidate sequence of primitives $P = (f_1, f_2, \dots, f_d)$, the candidate program is deterministically verified across all $K$ training demonstration pairs $\{(I_1, O_1), \dots, (I_K, O_K)\}$. The evaluation function measures exact pixel divergence:

$$\text{Loss}(P) = \sum_{k=1}^{K} \text{Divergence}(P(I_k), O_k)$$

* If the shape of the generated grid differs from the expected output shape, the loss is assigned as infinity ($\text{Loss} = \infty$).
* Otherwise, the loss corresponds to the exact count of mismatching pixels between prediction and target.

**Deterministic Acceptance Criterion:**
$$\text{Criterion:} \quad \text{Loss}(P^*) = 0 \quad (\text{100% exact match across all training pairs})$$

A program is accepted if and only if it achieves zero errors across every demonstration pair. The verified program is then applied to the hidden test input:

$$\text{Solved Test Grid} = P^*(I_{\text{test}})$$

---

## 3. Empirical Validation and Canonical Benchmarks (Accuracy & Completeness)

T-NSE underwent testing against both canonical ARC benchmarks and synthetic challenge suites:

| Test Case | Underlying Rule | Candidates Evaluated | Discovered Program | Result |
|---|---|:---:|:---:|:---:|
| **Official Task `007bbfb7`** | Kronecker fractal self-replication | 2 candidates | `fractal_self_tile` | **Exact ($\text{Loss}=0$)** |
| **Synthetic: 90° Rotation** | Pure clockwise rotation | 2 candidates | `rot90` | **Exact ($\text{Loss}=0$)** |
| **Synthetic: Horizontal Mirror** | Reflection along the vertical axis | 18 candidates | `flip_h` | **Exact ($\text{Loss}=0$)** |
| **Synthetic: Cavity Filling** | Enclosed hole filled with color 3 | 6 candidates | `fill_holes(3)` | **Exact ($\text{Loss}=0$)** |
| **Synthetic: Palette Recolor** | Global foreground recoloring to color 5 | 55 candidates | `recolor_all(5)` | **Exact ($\text{Loss}=0$)** |
| **Synthetic: Affine Composition** | 180° rotation followed by transposition | 9 candidates | `rot180 -> transpose` | **Exact (Depth 2)** |

### Offline Robustness in Kaggle ARC-AGI-2
Module 5 incorporates a submission pipeline engineered to satisfy competition constraints:
1. **Dynamic File Discovery:** Searches `/kaggle/input/` recursively via `glob` for `arc-agi_test_challenges.json`, ensuring independence from local directory naming conventions;
2. **Safe Execution Budget:** Employs a strict 1.5s per-task timeout quota, completing the evaluation of over 100 hidden tasks in under two minutes without timeout risks;
3. **Guaranteed Fallback & Strict Schema:** Produces valid `attempt_1` and `attempt_2` matrices formatted in standard JSON, preventing scoring errors caused by missing task keys.

---

## 4. Official Verification Proof (Submission ID)

In compliance with the official ARC Prize 2026 Paper Track rules (*Rule 2.1.b: team must match the team submitting to the main competition*), the implementation was evaluated on the official platform:

* **Associated Competition:** ARC Prize 2026 — ARC-AGI-2
* **Submission Name:** `T-NSE v6`
* **Official Verified Submission ID:** **`57107077`**
* **Evaluation Status:** Successfully scored and complete on the official leaderboard.

---

## 5. Universality Beyond ARC (Universality)

T-NSE is not an ad-hoc grid puzzle solver; it embodies a general framework for **Object-Centric Relational World Modeling**:

1. **Autonomous Robotics and Manipulation:** Inducing robotic arm trajectories from few visual demonstrations by converting perceptual video into spatial contact graphs and relative displacement vectors;
2. **Industrial Automation and CAD:** Verifying component clearances, inspecting PCB trace routing, and enforcing topological containment constraints;
3. **Physical Common-Sense Reasoning:** Predicting structural stability, container support, and occlusion dynamics using discrete topological invariants.

---

## 6. Honest Scope and Limitations (Limitations)

As a rigorous scientific contribution, we clearly delineate the current boundaries of our prototype:
* **DSL Expressivity:** The 75-primitive grammar excels at global geometric, affine, and topological fill operations, but tasks requiring continuous fluid simulation or complex multi-step arithmetic lie outside its present vocabulary;
* **Compositional Depth:** Deterministic search is bounded to depths $L \le 2$ to respect competition execution quotas;
* **Future Roadmap:** Ongoing work focuses on incorporating Graph Neural Networks (GNNs) over the Topological Scene Graph to probabilistically guide branch expansion, alongside object-level operators (`crop_bbox`, `isolate_object`, and directional gravity).

---

## 7. Conclusion

Achieving genuine fluid intelligence requires moving beyond the premise that all reasoning can be solved through statistical token interpolation. The **Topological Neuro-Symbolic Engine (T-NSE)** demonstrates that decoupling structured visual perception (grounded in physical objectness and discrete topology) from a deterministic program synthesis engine yields an explainable, robust, and sample-efficient approach to the Abstraction and Reasoning Corpus.

By enabling machines to "perceive" objects and their spatial relations before deducing transformation rules, T-NSE establishes a concrete bridge between perceptual representation and symbolic rigor.

---

### References

1. **Chollet, F.** (2019). *On the Measure of Intelligence*. arXiv preprint arXiv:1911.01547.
2. **Spelke, E. S., & Kinzler, K. D.** (2007). *Core knowledge*. Developmental Science, 10(1), 89-96.
3. **ARC Prize Foundation.** (2024–2026). *ARC-AGI-2 & ARC-AGI-3 Competitions and Technical Documentation*. arcprize.org.
4. **Santos, E. N. J.** (2026). *Topological Neuro-Symbolic Engine (T-NSE): Object-Centric Spatial Induction for ARC-AGI*. Official GitHub Repository and Kaggle Paper Track Submission.
