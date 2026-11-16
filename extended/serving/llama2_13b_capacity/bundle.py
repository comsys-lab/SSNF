"""Validate a full-length pilot, then measure capacity and throughput in one load."""
import math
import time
from protocol import save


def measure_bundle(llm, cfg, directory, caps, meter, run_trial):
    hf = llm.llm_engine.vllm_config.model_config.hf_config
    assert (hf.num_hidden_layers, hf.hidden_size, hf.num_attention_heads) == (40, 5120, 40)
    llm._experiment_meter = meter
    started = time.monotonic()
    counter = 0
    results = []
    cache = {}

    def trial(batch, mode, stage):
        nonlocal counter
        meter.assert_exclusive()
        counter += 1
        tag = f'{counter:03d}_{stage}_b{batch}'
        save(directory/'status.json', dict(state=stage, batch=batch, completed_trials=len(results), elapsed_s=time.monotonic()-started))
        local = dict(cfg, mode=mode)
        row = run_trial(llm, local, batch, directory/tag, tag)
        if row['complete_generation']:
            assert row['total_output_tokens'] == batch * cfg['output_tokens']
            assert math.isfinite(row['throughput']) and row['throughput'] > 0
        if mode == 'fixed':
            assert row['fixed_batch_valid']
            assert math.isclose(row['all_batch_decode_tok_s'], row['all_batch_decode_tokens']/row['all_batch_decode_wall_s'], rel_tol=1e-12)
        results.append(row)
        save(directory/'trials.json', results)
        return row

    pilot = trial(1, 'fixed', 'full_length_pilot')
    assert pilot['fixed_batch_valid']
    # A valid fixed-batch pilot also proves batch 1 remains resident to the end.
    cache[1] = pilot

    def capacity(batch):
        if batch in cache:
            return cache[batch]['complete_generation'] and cache[batch]['preemption_events'] == 0 and cache[batch]['full_batch_last_decode']
        row = trial(batch, 'capacity', 'capacity')
        success = bool(row['capacity_verified'])
        failure = bool(row.get('admission_blocked')) or (row['complete_generation'] and row['preemption_events'] > 0)
        if not (success or failure):
            raise RuntimeError('Unclassified capacity result: '+str(row))
        cache[batch] = row
        return success

    low = 1
    estimate = max(1, min(511, caps['conservative_batch']))
    if capacity(estimate):
        low = estimate
        high = estimate + 1
        while capacity(high):
            low = high
            high = min(512, max(high+1, high*2))
            if low == 512:
                raise RuntimeError('Capacity exceeds the declared max_num_seqs')
    else:
        high = estimate
    while high-low > 1:
        mid = (low+high)//2
        if capacity(mid):
            low = mid
        else:
            high = mid
    assert high == low + 1 and capacity(low) and not capacity(high)
    boundary = dict(maximum_successful_resident_batch=low, adjacent_failed_batch=high,
                    allocated_page_prediction=caps, success=cache[low], failure=cache[high],
                    policy='complete all input+256 output tokens; no preemption; all requests decode through final step')
    save(directory/'capacity_result.json', boundary)

    timings = []
    for batch in sorted({1, low}):
        trial(batch, 'fixed', 'timing_warmup')
        for repeat in range(3):
            row = trial(batch, 'fixed', 'throughput')
            row['timing_repeat'] = repeat
            timings.append(row)
            save(directory/'timing_trials.json', timings)
    meter.assert_exclusive()
    save(directory/'summary.json', dict(run_valid=True, stage='measure', mode='resident_capacity_and_fixed_throughput',
         format=cfg['format'], gpu_uuid=cfg['gpu_uuid'], model='Llama-2-13B',
         input_tokens=cfg['input_tokens'], output_tokens=cfg['output_tokens'],
         maximum_successful_resident_batch=low, adjacent_failed_batch=high,
         pilot=pilot, timing_trials=timings, all_trials=results,
         timing_repetitions='3 serial trials per batch in one model load; not independent reloads',
         elapsed_s=time.monotonic()-started, new_gpu_energy_claim=False))
    save(directory/'status.json', dict(state='complete', maximum_batch=low, timing_trials=len(timings)))
    return 0
