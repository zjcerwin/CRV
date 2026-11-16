"""Recompute the shared-response-model main results from prepared annotation data."""
import os

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import csv
from pathlib import Path
import numpy as np
from crv import calibrate, current_state, score, select, terminal_update

ROOT = Path(__file__).resolve().parent
DATASETS = ['wikitox', 'imagenet16h', 'dynasent_round1', 'dynasent_round2', 'cifar10h']
NAMES = {'plugin_value': 'CRV', 'random': 'Random', 'disagreement': 'Disagreement',
         'risk_only': 'Current risk', 'information_gain': 'Information gain'}


def reproduce(dataset):
    with np.load(ROOT / 'data' / (dataset + '.npz'), allow_pickle=False) as source:
        data = dict(source)
    # Fit from the included calibration annotations, then evaluate the fixed test histories.
    data['prior'], data['confusion'] = calibrate(data)
    posterior, stop = current_state(data)
    results = []
    for policy in NAMES:
        values = score(policy, data, posterior)
        selected = select(policy, values, data)
        # Candidate responses enter the decision only after selecting the purchase set.
        actions = terminal_update(data, posterior, stop, selected)
        errors = int(np.count_nonzero(actions != data['targets']))
        results.append({'dataset': dataset, 'policy': policy, 'n': len(actions),
                        'budget': int(data['budget']), 'errors': errors,
                        'terminal_error_percent': 100 * errors / len(actions)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, help='Run one dataset instead of all five')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/main_results.csv')
    args = parser.parse_args()
    with (ROOT / 'expected_results.csv').open(newline='', encoding='utf-8') as handle:
        expected = {(row['dataset'], row['policy']): row for row in csv.DictReader(handle)}
    results = []
    for dataset in [args.dataset] if args.dataset else DATASETS:
        batch = reproduce(dataset)
        for result in batch:
            reference = expected[(dataset, result['policy'])]
            for key in ['n', 'budget', 'errors']:
                if result[key] != int(reference[key]):
                    raise AssertionError(f'{dataset}/{result["policy"]}: {key} differs from the reference')
        print(dataset + ': ' + ', '.join(f'{NAMES[r["policy"]]} {r["terminal_error_percent"]:.2f}%' for r in batch))
        results.extend(batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f'{len(results)} evaluations match the reference error counts. Results: {args.output}')


if __name__ == '__main__':
    main()
