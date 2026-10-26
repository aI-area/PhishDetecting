"""Build the exact URL/label union used for descriptive corpus analyses."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, default=Path('your_path'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output path to preserve existing data.')
    frames = []
    for name in ('PhishFusion.csv', 'ebbu2017.csv', 'PhishStorm.csv'):
        frame = pd.read_csv(args.datasets/name, encoding='latin1')
        label = 'phishing' if 'phishing' in frame else 'label'
        frames.append(frame[['url', label]].rename(columns={label:'phishing'}))
    union = pd.concat(frames, ignore_index=True).drop_duplicates(['url','phishing'])
    if union.groupby('url').phishing.nunique().gt(1).any():
        raise ValueError('Conflicting labels require an explicit curation policy.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(args.output, index=False, encoding='latin1', lineterminator='\n')
    print(union.phishing.value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
