import numpy as np
from collections import deque

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # keeps the notebook runnable even without OpenCV
    HAS_CV2 = False

ARC_COLOR_MAP = {
    0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64),
    4: (255, 220, 0), 5: (170, 170, 170), 6: (240, 18, 190),
    7: (255, 133, 27), 8: (127, 219, 255), 9: (135, 12, 37),
}

def grid_to_rgb(grid):
    """Converts an ARC grid (values 0-9) into an RGB image for display."""
    grid = np.array(grid, dtype=np.uint8)
    rgb = np.zeros(grid.shape + (3,), dtype=np.uint8)
    for k, v in ARC_COLOR_MAP.items():
        rgb[grid == k] = v
    return rgb

print(f"[INFO] ARC palette ready | OpenCV available: {HAS_CV2}")

def label_components(binary_mask, connectivity=4):
    """Connected-component labeling. Uses OpenCV when available, pure-NumPy BFS otherwise.
    Returns (num_labels, labels) where label 0 is the background."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if HAS_CV2:
        n, labels = cv2.connectedComponents(np.uint8(binary_mask) * 255, connectivity=connectivity)
        return n, labels
    h, w = binary_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    n = 1
    for sy in range(h):
        for sx in range(w):
            if binary_mask[sy, sx] and labels[sy, sx] == 0:
                labels[sy, sx] = n
                q = deque([(sy, sx)])
                while q:
                    y, x = q.popleft()
                    for dy, dx in steps:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = n
                            q.append((ny, nx))
                n += 1
    return n, labels


class ARCObjectExtractor:
    def __init__(self, grid, connectivity=4, background=0):
        self.grid = np.array(grid, dtype=np.uint8)
        self.connectivity = connectivity   # 4 (orthogonal) or 8 (with diagonals)
        self.background = background
        self.objects = []

    def extract_features(self):
        """Isolates each color and extracts the spatial properties of every object."""
        self.objects = []
        for color in np.unique(self.grid):
            if color == self.background:
                continue
            mask = self.grid == color
            num_labels, labels = label_components(mask, self.connectivity)
            for i in range(1, num_labels):
                ys, xs = np.where(labels == i)
                x, y = int(xs.min()), int(ys.min())
                w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
                self.objects.append({
                    'color': int(color),
                    'bbox': (x, y, w, h),                          # (X, Y, Width, Height)
                    'area': int(len(xs)),                          # pixel mass
                    'centroid': (float(xs.mean()), float(ys.mean())),
                    'mask': np.uint8(labels == i),                 # exact shape
                })
        return self.objects


# --- Test: diagonal pixels, 4- vs 8-connectivity ---
diag_grid = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
for conn in (4, 8):
    objs = ARCObjectExtractor(diag_grid, connectivity=conn).extract_features()
    print(f"connectivity={conn}: {len(objs)} object(s)")

example_grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 2, 2, 2, 0],
    [0, 1, 1, 0, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
found_objects = ARCObjectExtractor(example_grid).extract_features()
for i, o in enumerate(found_objects):
    print(f"Object {i+1} | Color {o['color']} | Area {o['area']} | BBox {o['bbox']} | Centroid {o['centroid']}")

def enclosed_cells(mask):
    """Cells NOT in `mask` that cannot reach the grid border through non-mask cells (4-connected).
    These are the topological holes of the shape."""
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for y in range(h):
        for x in range(w):
            if (y in (0, h - 1) or x in (0, w - 1)) and not mask[y, x]:
                outside[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
    return (~mask.astype(bool)) & (~outside)


def build_scene_graph(objects, grid_shape):
    """Builds the Topological Scene Graph: nodes + typed edges."""
    nodes = [{'id': i, 'color': o['color'], 'area': o['area'], 'bbox': o['bbox'],
              'n_hole_cells': int(enclosed_cells(o['mask']).sum())} for i, o in enumerate(objects)]
    edges = []
    for i, a in enumerate(objects):
        holes_a = enclosed_cells(a['mask'])
        grown = np.zeros_like(a['mask'], dtype=bool)
        m = a['mask'].astype(bool)
        grown[1:, :] |= m[:-1, :]; grown[:-1, :] |= m[1:, :]
        grown[:, 1:] |= m[:, :-1]; grown[:, :-1] |= m[:, 1:]
        for j, b in enumerate(objects):
            if i == j:
                continue
            mb = b['mask'].astype(bool)
            delta = (b['centroid'][0] - a['centroid'][0], b['centroid'][1] - a['centroid'][1])
            if holes_a.any() and mb.any() and not (mb & ~holes_a).any():
                edges.append({'src': i, 'dst': j, 'relation': 'contains', 'delta': delta})
            elif (grown & mb).any():
                edges.append({'src': i, 'dst': j, 'relation': 'adjacent', 'delta': delta})
    return {'nodes': nodes, 'edges': edges}


tsg_grid = np.array([
    [3, 3, 3, 3, 3, 0, 4, 0],
    [3, 0, 0, 0, 3, 0, 4, 0],
    [3, 0, 5, 0, 3, 0, 0, 0],
    [3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 0, 0, 0],
])
tsg = build_scene_graph(ARCObjectExtractor(tsg_grid).extract_features(), tsg_grid.shape)
for n in tsg['nodes']:
    print("node", n)
for e in tsg['edges']:
    print("edge", e['src'], '->', e['dst'], e['relation'])

import itertools

# ---------- grid -> grid primitives ----------
def fill_holes(grid, color):
    g = np.array(grid, dtype=np.uint8).copy()
    holes = enclosed_cells(g != 0)
    g[holes] = color
    return g

def recolor_all(grid, color):
    g = np.array(grid, dtype=np.uint8).copy()
    g[g != 0] = color
    return g

def translate(grid, dx, dy):
    g = np.array(grid, dtype=np.uint8)
    out = np.zeros_like(g)
    h, w = g.shape
    ys, xs = np.where(g != 0)
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    out[ny[ok], nx[ok]] = g[ys[ok], xs[ok]]
    return out

def fractal_self_tile(grid):
    g = np.array(grid, dtype=np.uint8)
    return np.kron((g != 0).astype(np.uint8), g)

def upscale(grid, k):
    return np.kron(np.array(grid, dtype=np.uint8), np.ones((k, k), dtype=np.uint8))


class ARCSymbolicEngine:
    def __init__(self, extracted_objects=None, grid_shape=(10, 10)):
        self.objects = extracted_objects or []
        self.grid_shape = grid_shape
        self.primitives = self._build_dsl()

    # ---- object-level primitives (original rules, kept) ----
    def transform_recolor_all(self, target_color):
        new_grid = np.zeros(self.grid_shape, dtype=np.uint8)
        for obj in self.objects:
            new_grid[obj['mask'] > 0] = target_color
        return new_grid

    def transform_translate_object(self, obj_index, dx, dy):
        new_grid = np.zeros(self.grid_shape, dtype=np.uint8)
        for i, obj in enumerate(self.objects):
            if i == obj_index:
                ys, xs = np.where(obj['mask'] > 0)
                new_grid[np.clip(ys + dy, 0, self.grid_shape[0] - 1),
                         np.clip(xs + dx, 0, self.grid_shape[1] - 1)] = obj['color']
            else:
                new_grid[obj['mask'] > 0] = obj['color']
        return new_grid

    # ---- DSL ----
    def _build_dsl(self):
        P = {
            'identity': lambda g: np.array(g, dtype=np.uint8),
            'rot90':  lambda g: np.rot90(g, 1).copy(),
            'rot180': lambda g: np.rot90(g, 2).copy(),
            'rot270': lambda g: np.rot90(g, 3).copy(),
            'flip_h': lambda g: np.fliplr(g).copy(),
            'flip_v': lambda g: np.flipud(g).copy(),
            'transpose': lambda g: np.array(g).T.copy(),
            'fractal_self_tile': fractal_self_tile,
        }
        for c in range(1, 10):
            P[f'fill_holes({c})'] = (lambda g, c=c: fill_holes(g, c))
            P[f'recolor_all({c})'] = (lambda g, c=c: recolor_all(g, c))
        for k in (2, 3):
            P[f'upscale({k})'] = (lambda g, k=k: upscale(g, k))
        for dx, dy in itertools.product(range(-3, 4), repeat=2):
            if (dx, dy) != (0, 0):
                P[f'translate({dx},{dy})'] = (lambda g, dx=dx, dy=dy: translate(g, dx, dy))
        return P

    def _run(self, names, grid):
        g = np.array(grid, dtype=np.uint8)
        for n in names:
            g = self.primitives[n](g)
        return g

    def loss(self, names, train_pairs):
        """Total number of mismatching cells over ALL training pairs (shape mismatch = infinite)."""
        total = 0
        for inp, out in train_pairs:
            pred, out = self._run(names, inp), np.array(out, dtype=np.uint8)
            if pred.shape != out.shape:
                return float('inf')
            total += int((pred != out).sum())
        return total

    def prune_dsl(self, train_pairs):
        """Heuristic Pruning: Selects DSL families based on input/output shape and color differences."""
        allowed = [n for n in self.primitives if n != 'identity']
        shapes_in = [np.array(p[0]).shape for p in train_pairs]
        shapes_out = [np.array(p[1]).shape for p in train_pairs]
        same_shape = all(s_in == s_out for s_in, s_out in zip(shapes_in, shapes_out))
        transposed_shape = all(s_in[::-1] == s_out for s_in, s_out in zip(shapes_in, shapes_out))
        scaled_shapes = [(s_out[0]//s_in[0] if s_in[0]>0 else 0, s_out[1]//s_in[1] if s_in[1]>0 else 0) for s_in, s_out in zip(shapes_in, shapes_out)]
        is_scaled = all(sx == sy and sx in (2, 3) for sx, sy in scaled_shapes) and all(s_out[0] == s_in[0]*sx for s_in, s_out, (sx, _) in zip(shapes_in, shapes_out, scaled_shapes))
        is_fractal = all(s_out == (s_in[0]*s_in[0], s_in[1]*s_in[1]) for s_in, s_out in zip(shapes_in, shapes_out))

        to_keep = set()
        if same_shape:
            to_keep.update([n for n in allowed if n.startswith('translate') or n.startswith('fill_holes') or n.startswith('recolor_all') or n in ('rot180', 'flip_h', 'flip_v')])
            if all(s[0] == s[1] for s in shapes_in):
                to_keep.update(['rot90', 'rot270', 'transpose'])
        if transposed_shape:
            to_keep.update(['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'transpose'])
        if is_scaled:
            to_keep.update([n for n in allowed if n.startswith('upscale')])
        if is_fractal:
            to_keep.add('fractal_self_tile')
        if not to_keep:
            to_keep = set(allowed)

        out_colors = set(c for _, out in train_pairs for c in np.unique(out))
        return [n for n in to_keep if not (n.startswith('fill_holes') or n.startswith('recolor_all')) or int(n.split('(')[1].split(')')[0]) in out_colors]

    def synthesize(self, train_pairs, max_depth=2, max_time=5.0, max_progs=1, verbose=True):
        import time
        start_t = time.time()
        names = self.prune_dsl(train_pairs)
        if verbose:
            print(f"  [Pruner] Reduced DSL to {len(names)} primitives")
        tested = 0
        found = []
        for depth in range(1, max_depth + 1):
            for prog in itertools.product(names, repeat=depth):
                if time.time() - start_t > max_time:
                    if verbose:
                        print(f"  timeout after {tested} candidates")
                    return found
                tested += 1
                if self.loss(prog, train_pairs) == 0:
                    found.append(list(prog))
                    if verbose:
                        print(f"  program found: {' -> '.join(prog)}  (depth {depth}, {tested} candidates tested)")
                    if len(found) >= max_progs:
                        return found
        if verbose and not found:
            print(f"  no program found ({tested} candidates tested)")
        return found

    def solve(self, train_pairs, test_input, max_depth=2, max_time=5.0):
        progs = self.synthesize(train_pairs, max_depth, max_time, max_progs=2)
        preds = [self._run(p, test_input) for p in progs]
        return progs, preds

    # legacy single-example helper (kept for compatibility with the first prototype)
    def test_hypotheses(self, expected_output_grid):
        if np.array_equal(self.transform_recolor_all(3), expected_output_grid):
            return "RULE DISCOVERED: global recoloring to green."
        if np.array_equal(self.transform_translate_object(1, 0, 2), expected_output_grid):
            return "RULE DISCOVERED: translation of Object 2 by +2 on the Y axis."
        return "No basic rule matched. Use synthesize() for the full DSL."


# ---------- Sanity checks on synthetic multi-pair tasks ----------
rng = np.random.default_rng(0)
def rand_grid(h=5, w=5):
    return (rng.integers(0, 3, (h, w)) * rng.integers(0, 2, (h, w))).astype(np.uint8)

ring = np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]], dtype=np.uint8)
ring2 = np.array([[2,2,2,2,2],[2,0,0,0,2],[2,0,0,0,2],[2,2,2,2,2]], dtype=np.uint8)
synthetic_tasks = {
    "rotate 90":        [(g, np.rot90(g, 1)) for g in (rand_grid(4, 6), rand_grid(5, 5), rand_grid(3, 7))],
    "mirror horizontal": [(g, np.fliplr(g)) for g in (rand_grid(), rand_grid(4, 6))],
    "fill holes with 3": [(ring, fill_holes(ring, 3)), (ring2, fill_holes(ring2, 3))],
    "recolor to 5":     [(g, recolor_all(g, 5)) for g in (rand_grid(), rand_grid(4, 4))],
    "rotate 90 + flip": [(g, np.fliplr(np.rot90(g, 1))) for g in (rand_grid(4, 6), rand_grid(5, 5))],
}
engine = ARCSymbolicEngine()
for name, pairs in synthetic_tasks.items():
    print(f"[{name}]")
    engine.synthesize(pairs)

import glob, os, json, urllib.request

EMBEDDED_007BBFB7 = json.loads('''{"test":[{"input":[[7,0,7],[7,0,7],[7,7,0]],"output":[[7,0,7,0,0,0,7,0,7],[7,0,7,0,0,0,7,0,7],[7,7,0,0,0,0,7,7,0],[7,0,7,0,0,0,7,0,7],[7,0,7,0,0,0,7,0,7],[7,7,0,0,0,0,7,7,0],[7,0,7,7,0,7,0,0,0],[7,0,7,7,0,7,0,0,0],[7,7,0,7,7,0,0,0,0]]}],"train":[{"input":[[0,7,7],[7,7,7],[0,7,7]],"output":[[0,0,0,0,7,7,0,7,7],[0,0,0,7,7,7,7,7,7],[0,0,0,0,7,7,0,7,7],[0,7,7,0,7,7,0,7,7],[7,7,7,7,7,7,7,7,7],[0,7,7,0,7,7,0,7,7],[0,0,0,0,7,7,0,7,7],[0,0,0,7,7,7,7,7,7],[0,0,0,0,7,7,0,7,7]]},{"input":[[4,0,4],[0,0,0],[0,4,0]],"output":[[4,0,4,0,0,0,4,0,4],[0,0,0,0,0,0,0,0,0],[0,4,0,0,0,0,0,4,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,4,0,4,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,4,0,0,0,0]]},{"input":[[0,0,0],[0,0,2],[2,0,2]],"output":[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,2],[0,0,0,0,0,0,2,0,2],[0,0,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0,2],[2,0,2,0,0,0,2,0,2]]},{"input":[[6,6,0],[6,0,0],[0,6,6]],"output":[[6,6,0,6,6,0,0,0,0],[6,0,0,6,0,0,0,0,0],[0,6,6,0,6,6,0,0,0],[6,6,0,0,0,0,0,0,0],[6,0,0,0,0,0,0,0,0],[0,6,6,0,0,0,0,0,0],[0,0,0,6,6,0,6,6,0],[0,0,0,6,0,0,6,0,0],[0,0,0,0,6,6,0,6,6]]},{"input":[[2,2,2],[0,0,0],[0,2,2]],"output":[[2,2,2,2,2,2,2,2,2],[0,0,0,0,0,0,0,0,0],[0,2,2,0,2,2,0,2,2],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,2,2,2,2,2,2],[0,0,0,0,0,0,0,0,0],[0,0,0,0,2,2,0,2,2]]}]}''')

def load_arc_task(task_id="007bbfb7"):
    """Loads an ARC task: local Kaggle input -> GitHub (with timeout) -> embedded copy."""
    patterns = [f"/kaggle/input/**/{task_id}.json", f"./**/{task_id}.json"]
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    print(f"[source] local file: {path}")
                    return json.load(f)
            except (OSError, ValueError):
                continue
    url = f"https://raw.githubusercontent.com/fchollet/ARC/master/data/training/{task_id}.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            print(f"[source] downloaded from GitHub: {url}")
            return json.loads(resp.read())
    except Exception as exc:  # no Internet / DNS / timeout / HTTP error
        print(f"[warning] download failed ({type(exc).__name__}); using embedded copy")
        if task_id == "007bbfb7":
            return EMBEDDED_007BBFB7
        raise

task_data = load_arc_task("007bbfb7")
print(f"Training pairs: {len(task_data['train'])} | Test pairs: {len(task_data['test'])}")

# --- Perception on real data ---
real_input = task_data['train'][0]['input']
objs = ARCObjectExtractor(real_input, connectivity=4).extract_features()
print(f"Objects (4-conn): {len(objs)} | (8-conn): {len(ARCObjectExtractor(real_input, connectivity=8).extract_features())}")

# --- Full pipeline: synthesize on ALL train pairs, then predict the hidden test output ---
train_pairs = [(p['input'], p['output']) for p in task_data['train']]
test_in, test_out = task_data['test'][0]['input'], np.array(task_data['test'][0]['output'])
engine = ARCSymbolicEngine()
progs, preds = engine.solve(train_pairs, test_in)
print("Synthesized programs:", progs)
print("Test prediction correct:", bool(len(preds) > 0 and np.array_equal(preds[0], test_out)))

import os
import json

# Paths for Kaggle ARC-AGI-2 Environment
KAGGLE_TEST_DIR = "/kaggle/input/arc-prize-2026-arc-agi-2/test/"
KAGGLE_TEST_FILE = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"

submission_dict = {}
test_tasks = {}

# 1. Load the test data
if os.path.exists(KAGGLE_TEST_FILE):
    with open(KAGGLE_TEST_FILE, 'r') as f:
        test_tasks = json.load(f)
elif os.path.exists(KAGGLE_TEST_DIR):
    for f_name in os.listdir(KAGGLE_TEST_DIR):
        if f_name.endswith('.json'):
            task_id = f_name.split('.')[0]
            with open(os.path.join(KAGGLE_TEST_DIR, f_name), 'r') as f:
                test_tasks[task_id] = json.load(f)
else:
    print("Kaggle test data not found. Falling back to the embedded task for demonstration.")
    # Fallback to the task loaded in Module 4 for local validation
    try:
        test_tasks = {"007bbfb7": EMBEDDED_007BBFB7}
    except NameError:
        print("Embedded task not found either.")

# 2. Iterate and generate predictions
for task_id, task_data in test_tasks.items():
    if 'train' in task_data and 'test' in task_data:
        # Format train pairs as expected by ARCSymbolicEngine (list of (input, output) tuples)
        train_pairs = [(p['input'], p['output']) for p in task_data['train']]
        test_pairs = task_data['test']

        # Instantiate engine
        engine = ARCSymbolicEngine()

        # One entry per test input (a few tasks have 2 or more test inputs)
        task_entries = []
        for t_pair in test_pairs:
            test_input = t_pair['input']
            try:
                # Depth 2 is now safe with pruning and a 4.0s timeout per task
                progs, preds = engine.solve(train_pairs, test_input, max_depth=2, max_time=4.0)
            except Exception as e:
                print(f"[{task_id}] solver error: {e}")
                preds = []

            attempt_1 = attempt_2 = np.array(test_input).tolist()
            if len(preds) > 0:
                attempt_1 = np.array(preds[0]).tolist()
                attempt_2 = np.array(preds[-1]).tolist()

            task_entries.append({"attempt_1": attempt_1, "attempt_2": attempt_2})

        # Format required by the Kaggle ARC-AGI-2 competition:
        # {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]} - one dict per test input
        if len(task_entries) > 0:
            submission_dict[task_id] = task_entries

# 3. Save submission.json (Kaggle expects it in the working directory)
submission_dir = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
submission_path = os.path.join(submission_dir, 'submission.json')
with open(submission_path, 'w') as f:
    json.dump(submission_dict, f)

print(f"Submission generated at {os.path.abspath(submission_path)} with {len(submission_dict)} tasks.")
