"""Build the selected KV comparison from immutable, fully aligned raw results.

Default policy excludes FP16 outlier preservation from primary KVQuant rows.
Historical 1% results remain explicitly labeled in a secondary table.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ACCURACY = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ACCURACY/'kv_comparison_selection.json'
DEFAULT_OUTPUT = ACCURACY/'analysis/output/kv_cache_selected'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(selection):
    tables = {'primary': [], 'secondary': []}
    reference = None
    reference_meta = None
    evidence = {}
    for section, table in tables.items():
        for item in selection[section]:
            folder = ACCURACY/item['folder']
            meta, summary = read(folder/'meta.json'), read(folder/'summary.json')
            if section == 'primary' and meta['method'].startswith('kvquant'):
                if meta['settings']['sparse'] != selection['kvquant_outlier_fraction']:
                    raise ValueError('primary KVQuant must disable FP16 outlier preservation')
            raw = sorted((json.loads(x) for x in (folder/'windows.jsonl').read_text().splitlines()), key=lambda x: x['window_id'])
            assert summary['complete'] and summary['full_test']
            assert len(raw) == len(meta['windows']) == 330
            assert [x['window_id'] for x in raw] == list(range(330))
            for row, expected in zip(raw, meta['windows']):
                assert all(row[k] == v for k, v in expected.items())
                assert row['protocol_hash'] == meta['protocol_hash'] == summary['protocol_hash']
                assert math.isfinite(row['nll']) and row['nll'] > 0
            count = sum(x['target_count'] for x in raw)
            nll = math.fsum(x['nll'] for x in raw)
            ppl = math.exp(nll/count)
            assert count == summary['target_count'] == 338534
            assert math.isclose(ppl, summary['ppl'], rel_tol=1e-12)
            if reference is None:
                reference, reference_meta = raw, meta
            for k in ('inputs','token_sha256','context','stride','cache_reset','position_ids','windows'):
                assert meta[k] == reference_meta[k], k
            for a, b in zip(reference, raw):
                assert a['input_sha256'] == b['input_sha256']
            table.append(dict(method=item['label'], weight_activation_dtype=meta['weight_activation_dtype'],
                outlier_fraction=meta['settings'].get('sparse', ''), ppl=ppl,
                target_count=count, windows=len(raw), raw_results=str(folder)))
            evidence[item['folder']] = {n:sha(folder/n) for n in ('meta.json','summary.json','windows.jsonl')}
    bf16 = tables['primary'][0]['ppl']
    ssnf = next(x['ppl'] for x in tables['primary'] if x['method']=='SSNF KV-only')
    for table in tables.values():
        for row in table:
            row['delta_bf16'] = row['ppl']-bf16
            row['ssnf_minus_method_ppl'] = ssnf-row['ppl']
    return dict(selection=selection, tables=tables, audit=dict(passed=True, aligned_windows=330,
        unique_targets=338534, primary_kvquant_outliers_disabled=True, evidence=evidence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    selection = read(args.selection)
    assert selection['kvquant_outlier_fraction'] == 0
    result = assemble(selection)
    args.output.mkdir(parents=True, exist_ok=True)
    for section, table in result['tables'].items():
        with (args.output/f'{section}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
    result['selection_sha256'] = sha(args.selection)
    result['report_source_sha256'] = sha(Path(__file__))
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['tables']['primary'], indent=2))


if __name__ == '__main__':
    main()
