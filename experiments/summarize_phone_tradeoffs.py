"""Recompute mean-coordinate detection/resource frontiers from retained tables."""
import argparse
import csv
from pathlib import Path


def membership(rows, keys):
    output = []
    for row in rows:
        def coordinates(r):
            return [(-1 if key in ('accuracy', 'f1') else 1)*float(r[key]) for key in keys]
        point = coordinates(row)
        dominators = []
        for other in rows:
            values = coordinates(other)
            if all(a <= b for a, b in zip(values, point)) and any(a < b for a, b in zip(values, point)):
                dominators.append(other['model'])
        output.append(dict(model=row['model'], objectives=';'.join(keys),
                           non_dominated=not dominators, dominated_by=';'.join(dominators)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.results/'phone_quality_resources.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['batch_size']) == 1]
    records = []
    for quality in ('accuracy', 'f1'):
        for energy in ('gross_mj', 'incremental_mj'):
            records.extend(membership(rows, (quality, 'fpr', 'latency_ms', 'peak_rss_mib', energy)))
    with (args.output/'pareto_membership.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print('Recomputed four five-objective mean-coordinate comparisons; no measurements or predictions changed.')


if __name__ == '__main__':
    main()
