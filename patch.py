import json
import numpy as np

file_path = 'D:\\NEURO_SIMBOLIC_ESPACILA_ARC\\Neuro-Simbolic_Espacial_Topologico_ARC.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Module 3 updates
        if 'def synthesize(self, train_pairs, max_depth=2, verbose=True):' in source:
            new_methods = '''    def prune_dsl(self, train_pairs):
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
            to_keep.update(['rot90', 'rot270', 'transpose'])
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
'''
            old_methods = '''    def synthesize(self, train_pairs, max_depth=2, verbose=True):
        """Shortest-first search; returns the first program with Loss == 0 on every training pair."""
        names = [n for n in self.primitives if n != 'identity']
        tested = 0
        for depth in range(1, max_depth + 1):
            for prog in itertools.product(names, repeat=depth):
                tested += 1
                if self.loss(prog, train_pairs) == 0:
                    if verbose:
                        print(f"  program found: {' -> '.join(prog)}  (depth {depth}, {tested} candidates tested)")
                    return list(prog)
        if verbose:
            print(f"  no program found ({tested} candidates tested)")
        return None

    def solve(self, train_pairs, test_input, max_depth=2):
        prog = self.synthesize(train_pairs, max_depth)
        return (prog, self._run(prog, test_input)) if prog else (None, None)
'''
            source = source.replace(old_methods, new_methods)
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

        # Module 4 updates
        if 'program, prediction = engine.solve(train_pairs, test_in)' in source:
            source = source.replace('program, prediction = engine.solve(train_pairs, test_in)', 'progs, preds = engine.solve(train_pairs, test_in)')
            source = source.replace('print("Synthesized program:", program)', 'print("Synthesized programs:", progs)')
            source = source.replace('print("Test prediction correct:", bool(prediction is not None and np.array_equal(prediction, test_out)))', 'print("Test prediction correct:", bool(len(preds) > 0 and np.array_equal(preds[0], test_out)))')
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

        # Module 5 updates
        if 'program, pred = engine.solve(train_pairs, test_input, max_depth=0)' in source:
            old_loop = '''                # Use the solve helper which synthesizes and runs
                program, pred = engine.solve(train_pairs, test_input, max_depth=0)
            except Exception as e:
                # One failing task must never break the whole submission
                print(f"[{task_id}] solver error: {e}")
                pred = None

            if pred is not None:
                attempt_1 = np.array(pred).tolist()
            else:
                # Default guess if synthesis fails: copy the test input
                attempt_1 = np.array(test_input).tolist()

            # Prototype has a single hypothesis, so both attempts are identical
            task_entries.append({"attempt_1": attempt_1, "attempt_2": attempt_1})'''
            new_loop = '''                # Depth 2 is now safe with pruning and a 4.0s timeout per task
                progs, preds = engine.solve(train_pairs, test_input, max_depth=2, max_time=4.0)
            except Exception as e:
                print(f"[{task_id}] solver error: {e}")
                preds = []

            attempt_1 = attempt_2 = np.array(test_input).tolist()
            if len(preds) > 0:
                attempt_1 = np.array(preds[0]).tolist()
                attempt_2 = np.array(preds[-1]).tolist()

            task_entries.append({"attempt_1": attempt_1, "attempt_2": attempt_2})'''
            source = source.replace(old_loop, new_loop)
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('done')
