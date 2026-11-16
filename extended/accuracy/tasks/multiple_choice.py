"""Core eight-task prompts on the common HF quantization engine.

This is the core prompt/scorer family, not an external-harness score.
The causal pair boundary follows TemplateLM._encode_pair, with special tokens
explicitly disabled for task prompts. Raw likelihoods and token counts are saved.
Reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import torch
import torch.nn.functional as F

from engine import ppl
from engine import ppl_matched

EXPECTED = {'piqa': 1838, 'arc_easy': 2376, 'arc_challenge': 1172, 'hellaswag': 10042,
            'winogrande': 1267, 'boolq': 3270, 'openbookqa': 500, 'mmlu': 14042}
VERSION = 'core-mc-boundary-v1'


def parse():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--tasks', default=','.join(EXPECTED))
    extra.add_argument('--task-limit', type=int, default=0, help='0=full; otherwise nested spread across complete split.')
    extra.add_argument('--doc-batch', type=int, default=8)
    extra.add_argument('--candidate-batch', type=int, default=8)
    extra.add_argument('--batch-token-budget', type=int, default=4096)
    extra.add_argument('--task-max-length', type=int, default=8192)
    extra.add_argument('--documents-dir', default=None, help='Optional cached core-built task JSON documents.')
    extra.add_argument('--export-documents-only', action='store_true')
    extra.add_argument('--frozen-map-dir', default=None)
    x, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    a = ppl.parse()
    if min(x.doc_batch, x.candidate_batch, x.batch_token_budget, x.task_max_length) < 1:
        raise ValueError('batch and length values must be positive')
    if any(t not in EXPECTED for t in x.tasks.split(',')):
        raise ValueError('unknown multiple-choice task')
    return a, x


def load_builder(core):
    path = Path(core) / 'eval_accuracy/eval_zeroshot.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('core_zeroshot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build, path


def documents(task, builder, directory):
    path = Path(directory) / (task + '.json') if directory else None
    if path and path.exists():
        saved = json.loads(path.read_text())
        docs = saved['documents']
        if saved['documents_sha256'] != ppl.digest(docs):
            raise RuntimeError('task document checksum mismatch')
    else:
        docs = builder(task)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            ppl.atomic_json(path, {'task': task, 'documents': docs, 'documents_sha256': ppl.digest(docs)})
    if len(docs) != EXPECTED[task]:
        raise RuntimeError(f'{task}: expected complete split {EXPECTED[task]}, found {len(docs)}; investigate dataset snapshot')
    return docs


def encode_pair(tok, context, continuation):
    if not continuation:
        raise ValueError('empty continuation cannot silently become an EOS target')
    encode = lambda text: list(tok(text, add_special_tokens=False).input_ids)
    if not context:
        target = encode(continuation)
        prefix = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        if prefix is None:
            raise ValueError('empty context needs an explicit BOS/EOS conditioning token')
        context_ids = [prefix]
        if target and target[0] == prefix:
            context_ids, target = target[:1], target[1:]
        whole, separated_different = context_ids + target, False
    else:
        n = len(context) - len(context.rstrip())
        if n:
            context, continuation = context[:-n], context[-n:] + continuation
        context_ids = encode(context)
        whole = encode(context + continuation)
        target = whole[len(context_ids):]
        separated_different = target != encode(continuation)
    if not context_ids or not target:
        raise ValueError('pair encoding produced no conditioning or target tokens')
    # Record the prefix discrepancy rather than silently claiming concatenation equivalence.
    return context_ids, target, {'separate_continuation_differs': separated_different,
        'whole_prefix_matches_context': whole[:len(context_ids)] == context_ids}


@torch.no_grad()
def score_pairs(engine, pairs, options):
    encoded = []
    unique = {}
    for context, continuation in pairs:
        c, k, boundary = encode_pair(engine.tok, context, continuation)
        inp = c + k[:-1]
        if len(inp) > options.task_max_length:
            raise RuntimeError(f'task input {len(inp)} exceeds frozen max {options.task_max_length}; no silent truncation')
        key = tuple(inp)
        unique.setdefault(key, []).append(len(encoded))
        encoded.append({'input_ids': inp, 'target_ids': k, 'target_start': len(c)-1,
            'context_tokens': len(c), 'target_count': len(k), 'boundary': boundary})
    keys = sorted(unique, key=lambda k: (-len(k), k))
    batches, batch, largest = [], [], 0
    for key in keys:
        length = max(largest, len(key))
        if batch and (len(batch) >= options.candidate_batch or (len(batch)+1)*length > options.batch_token_budget):
            batches.append(batch); batch, largest = [], 0
        batch.append(key); largest = max(largest, len(key))
    if batch:
        batches.append(batch)
    results = [None] * len(pairs)
    for batch in batches:
        if engine.should_stop():
            raise TimeoutError('task budget exhausted before next forward')
        pad = engine.tok.pad_token_id
        if pad is None:
            pad = engine.tok.eos_token_id or 0
        ids = torch.full((len(batch), max(map(len, batch))), pad, dtype=torch.long, device=engine.device)
        mask = torch.zeros_like(ids)
        for i, key in enumerate(batch):
            ids[i, :len(key)] = torch.tensor(key, device=engine.device)
            mask[i, :len(key)] = 1
        engine.calls = {}
        logits = engine.model(ids, attention_mask=mask, use_cache=False).logits
        if engine.scope and (len(engine.calls) != len(engine.layers) or any(n != 1 for n in engine.calls.values())):
            raise RuntimeError('task KV quantization did not execute exactly once per layer')
        for i, key in enumerate(batch):
            for index in unique[key]:
                row = encoded[index]
                n, start = row['target_count'], row['target_start']
                scores = logits[i, start:start+n].float()
                targets = torch.tensor(row['target_ids'], device=engine.device)
                if len(scores) != n:
                    raise RuntimeError('task scorer target/logit count mismatch')
                ll = -F.cross_entropy(scores, targets, reduction='none').double().sum().item()
                if not math.isfinite(ll):
                    raise RuntimeError('nonfinite task likelihood')
                results[index] = {'loglikelihood': ll, 'target_count': n, 'context_tokens': row['context_tokens'],
                    'input_tokens': len(key), 'input_sha256': ppl.digest(list(key)), 'target_sha256': ppl.digest(row['target_ids']),
                    **row['boundary']}
        del logits, ids, mask
    assert all(r is not None for r in results)
    return results, {'unique_forward_sequences': len(unique), 'candidate_requests': len(pairs), 'forward_batches': len(batches)}


def summary(rows, selected, full_count, method, task, protocol_hash):
    chosen = [rows[i] for i in selected if i in rows]
    n = len(chosen)
    return {'status': 'complete' if n == len(selected) else 'partial', 'task': task, 'method': method,
        'protocol_hash': protocol_hash, 'completed_examples': n, 'requested_examples': len(selected),
        'full_split_examples': full_count, 'evaluation_extent': 'full' if len(selected)==full_count else 'prespecified_corpus_spread_subset',
        'acc': 100*sum(r['correct'] for r in chosen)/n if n else None,
        'acc_norm': 100*sum(r['correct_norm'] for r in chosen)/n if n else None,
        'target_count': sum(sum(c['target_count'] for c in r['candidates']) for r in chosen),
        'example_ids': [r['example_id'] for r in chosen], 'updated_unix': time.time()}


def evaluate_task(engine, options, method, task, docs, builder_path):
    out = Path(engine.a.out) / method / task
    out.mkdir(parents=True, exist_ok=True)
    order = ppl.spread_order(len(docs))
    selected = order if options.task_limit == 0 else order[:options.task_limit]
    base, _ = engine.protocol(method)
    meta = {'task_version': VERSION, 'task_engine_sha256': ppl.sha_file(Path(__file__)),
        'core_prompt_builder_sha256': ppl.sha_file(builder_path), 'ppl_engine': base,
        'task': task, 'documents_sha256': ppl.digest(docs), 'all_example_order': order,
        'tokenization': 'TemplateLM causal pair boundary; explicit add_special_tokens=False; right padding; no truncation',
        'primary_metric': 'argmax summed continuation log likelihood; auxiliary normalization by original continuation characters',
        'score_identity': 'core eight-task prompts with new explicit boundary handling; not external-harness score',
        'task_max_length': options.task_max_length}
    protocol_hash = ppl.digest(meta)
    old = out / 'meta.json'
    if old.exists() and json.loads(old.read_text())['protocol_hash'] != protocol_hash:
        raise RuntimeError('task resume protocol mismatch')
    ppl.atomic_json(old, dict(meta, protocol_hash=protocol_hash, selected_example_ids=selected, requested_examples=len(selected)))
    rows = {}
    log = out / 'examples.jsonl'
    if log.exists():
        for line in log.read_text().split('\n'):
            if not line:
                continue
            row = json.loads(line)
            if row['protocol_hash'] != protocol_hash or row['example_id'] in rows:
                raise RuntimeError('mixed/duplicate task checkpoint records')
            rows[row['example_id']] = row
    todo = [i for i in selected if i not in rows]
    t0 = time.time()
    for offset in range(0, len(todo), options.doc_batch):
        if engine.should_stop():
            break
        indices = todo[offset:offset+options.doc_batch]
        pairs, lookup = [], []
        for index in indices:
            context, continuation, gold = docs[index]
            if task == 'winogrande':
                pp = [(c, continuation) for c in context]
            else:
                pp = [(context, c) for c in continuation]
            lookup.append((index, len(pairs), len(pp), gold, [max(1,len(c)) for _,c in pp]))
            pairs.extend(pp)
        bt = time.time()
        try:
            scores, audit = score_pairs(engine, pairs, options)
        except TimeoutError:
            break
        with log.open('a') as f:
            for index, start, count, gold, lengths in lookup:
                candidates = scores[start:start+count]
                values = [c['loglikelihood'] for c in candidates]
                prediction = max(range(count), key=lambda j: values[j])
                prediction_norm = max(range(count), key=lambda j: values[j]/lengths[j])
                row = {'record': 'example', 'task': task, 'model': engine.model_id, 'method': method,
                    'protocol_hash': protocol_hash, 'example_id': index, 'document_sha256': ppl.digest(docs[index]),
                    'gold': gold, 'prediction': prediction, 'prediction_norm': prediction_norm,
                    'correct': prediction==gold, 'correct_norm': prediction_norm==gold,
                    'continuation_characters': lengths, 'candidates': candidates,
                    'completed_unix': time.time(), 'batch_seconds': time.time()-bt, 'batch_audit': audit}
                f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
                rows[index] = row
        s = summary(rows, selected, len(docs), method, task, protocol_hash)
        s['elapsed_s'] = time.time()-t0
        ppl.atomic_json(out/'status.json', s)
        print(json.dumps({'event': 'task_progress', 'task':task,'method':method,
            'completed_examples':s['completed_examples'],'requested_examples':len(selected),
            'acc':s['acc'],'seconds':time.time()-t0,
            'examples_per_second':(offset+len(indices))/max(time.time()-t0,1e-9)}), flush=True)
    s = summary(rows, selected, len(docs), method, task, protocol_hash)
    s['elapsed_s'] = time.time()-t0
    ppl.atomic_json(out/'summary.json', s)
    return s


def main():
    a, options = parse()
    if not a.core_root:
        raise ValueError('--core-root is needed for the exact core prompts')
    if a.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_DATASETS_OFFLINE'] = '1'
    builder, builder_path = load_builder(a.core_root)
    all_docs = {task: documents(task, builder, options.documents_dir) for task in options.tasks.split(',')}
    if options.export_documents_only:
        return
    if a.threads:
        torch.set_num_threads(a.threads)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, ppl.request_stop)
    ppl_matched.MAP_DIR = options.frozen_map_dir
    engine = None
    started = time.time()
    try:
        engine = ppl_matched.MatchedEngine(a, started)
        methods = a.methods.split(',') if a.methods else [a.method]
        for method in methods:
            if engine.should_stop():
                break
            out = Path(a.out)/method; out.mkdir(parents=True, exist_ok=True)
            pending = False
            for task, docs in all_docs.items():
                existing = out/task/'summary.json'
                old = json.loads(existing.read_text()) if existing.exists() else {}
                requested = min(len(docs), options.task_limit) if options.task_limit else len(docs)
                pending |= old.get('status') != 'complete' or old.get('requested_examples') != requested
            if pending:
                t0 = time.time(); engine.prepare(method)
                ppl.atomic_json(out/'encoding.json', {'method':method,'seconds':time.time()-t0,'storage':engine.storage,
                    'weight_tensors':engine.enc_details,'map_calibration':getattr(engine,'map_calibration',None)})
            results = []
            for task, docs in all_docs.items():
                if engine.should_stop():
                    break
                s = evaluate_task(engine, options, method, task, docs, builder_path)
                results.append(s)
                if s['status'] != 'complete':
                    break
            complete = len(results)==len(all_docs) and all(r['status']=='complete' for r in results)
            ppl.atomic_json(out/'suite_summary.json', {'method':method,'tasks':results,
                'macro_acc':sum(r['acc'] for r in results)/len(results) if results and complete else None,
                'requested_tasks':options.tasks.split(','),'completed_tasks':sum(r['status']=='complete' for r in results),
                'status':'complete' if complete else 'partial'})
    finally:
        if engine:
            engine.close()


if __name__ == '__main__':
    main()
