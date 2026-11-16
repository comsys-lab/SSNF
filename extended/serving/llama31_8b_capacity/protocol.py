"""CPU-only protocol and validation for fixed-batch/capacity experiments."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def page_capacity(total_pages, page_tokens, input_tokens, output_tokens):
    # Last sampled output does not itself need a KV slot. We reserve it anyway
    # for the declared input+output capacity, matching the paper's convention.
    per_request = math.ceil((input_tokens + output_tokens) / page_tokens)
    return dict(total_pages=total_pages, page_tokens=page_tokens,
                reserved_pages_per_request=per_request,
                conservative_batch=total_pages // per_request,
                last_forward_pages_per_request=math.ceil((input_tokens+output_tokens-1)/page_tokens))


class PrefillBarrier:
    """Use the scheduler's existing decode eligibility gate; never edit tokens/KV.

    The root fixes this protocol before measurement. It is a diagnostic scheduling
    policy, not deployment TTFT. Original eligible values are restored afterwards.
    """
    def __init__(self, scheduler, request_ids):
        self.scheduler = scheduler
        self.request_ids = set(request_ids)
        self.released = False
        self.original = {}

    def before_schedule(self):
        reqs = self.scheduler.requests
        ready = all(rid in reqs and reqs[rid].num_computed_tokens >= reqs[rid].num_prompt_tokens
                    and reqs[rid].num_output_tokens >= 1 for rid in self.request_ids)
        if ready:
            self.released = True
        for rid in self.request_ids:
            req = reqs.get(rid)
            if req is None:
                continue
            if self.released:
                if rid in self.original:
                    req.next_decode_eligible_step = self.original.pop(rid)
            elif req.num_computed_tokens >= req.num_prompt_tokens:
                self.original.setdefault(rid, req.next_decode_eligible_step)
                req.next_decode_eligible_step = 10**15
        return self.released

    def restore(self):
        for rid, value in self.original.items():
            if rid in self.scheduler.requests:
                self.scheduler.requests[rid].next_decode_eligible_step = value
        self.original.clear()


def validate_trial(rows, steps, batch, output_tokens, mode):
    issues = []
    complete = len(rows) == batch and all(r.get('finished') and r['output_tokens'] == output_tokens for r in rows)
    if not complete:
        issues.append('incomplete_generation')
    if any(not math.isfinite(s['wall_s']) or s['wall_s'] <= 0 for s in steps):
        issues.append('invalid_step_time')
    ids = {r['request_id'] for r in rows}
    pure = [s for s in steps if set(s['scheduled_ids']) == ids and
            set(s['decode_ids']) == ids and all(n == 1 for n in s['scheduled_tokens'].values())]
    preemptions = sum(len(s['preempted']) for s in steps)
    final_full = [s for s in pure if s['min_computed_before'] >= s['input_tokens']+output_tokens-2]
    decode_seconds = sum(s['wall_s'] for s in pure)
    record = dict(complete_generation=complete, preemption_events=preemptions,
                  all_batch_decode_steps=len(pure), last_full_batch_steps=len(final_full),
                  all_batch_decode_wall_s=decode_seconds,
                  all_batch_decode_tokens=len(pure)*batch,
                  all_batch_decode_tok_s=len(pure)*batch/decode_seconds if decode_seconds else None,
                  full_batch_last_decode=bool(final_full),
                  issues=issues)
    if mode == 'fixed':
        # Async scheduler may emit an empty final drain step. It computes no
        # tokens and is not a decode step; retain it in raw/end-to-end wall time.
        after = [s for s in steps if s['barrier_released'] and s['scheduled_ids']]
        stable = (len(after) == output_tokens-1 and len(pure) == len(after)
                  and all(s in pure for s in after) and preemptions == 0)
        record['fixed_batch_valid'] = complete and stable and not issues
        if not stable:
            issues.append('decode_not_constant_for_all_remaining_tokens')
    else:
        record['capacity_verified'] = complete and preemptions == 0 and bool(final_full) and not issues
    return record
