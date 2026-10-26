"""Profile candidate shift cohorts and source overlap without fitting models."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
import pandas as pd
import tldextract

EXTRACT=tldextract.TLDExtract(suffix_list_urls=(),include_psl_private_domains=True)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()


def parse(value):
    value=html.unescape(str(value).strip())
    try:
        p=urlsplit(value if '://' in value else '//'+value)
        host=(p.hostname or '').lower().rstrip('.')
        ext=EXTRACT(host)
        domain=getattr(ext,'top_domain_under_public_suffix',None) or getattr(ext,'registered_domain','') or host
        # Keep path/query case and content; ignore scheme and fragment for overlap.
        normalized=urlunsplit(('',host+((':'+str(p.port)) if p.port else ''),p.path or '/',p.query,''))
        return normalized,domain,host,bool(p.scheme in ('http','https') and host),bool(p.path.strip('/') or p.query)
    except (ValueError,TypeError):
        return '', '', '', False,False


def load(path):
    try: return pd.read_csv(path)
    except UnicodeDecodeError: return pd.read_csv(path,encoding='latin1')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--source',type=Path,default=Path('your_path/PhishFusion.csv'))
    p.add_argument('--snapshot',type=Path,default=Path('your_path/snapshot.csv'))
    p.add_argument('--phishtank',type=Path,default=Path('your_path/Phishtank/PhishTank.csv'))
    p.add_argument('--reference-root',type=Path,default=Path('your_path/reference_experiments'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts/litephish'))
    args=p.parse_args()
    assert not args.output.exists(), 'Do not overwrite audit evidence'
    args.output.mkdir(parents=True)
    base=Path('your_path')
    source_path=args.source
    source=load(source_path)
    src=[parse(u) for u in source['url'].fillna('')]
    source_urls={r[0] for r in src if r[0]}
    source_domains={r[1] for r in src if r[1]}
    datasets={
        'LitePhish_source_PhishFusion':source_path,
        'ExpertFusion_TE6':args.snapshot,
        'ExpertFusion_TE2':base/'optional_Phishtank.csv',
        'ExpertFusion_TE3':base/'optional_StealthPhisher.csv',
        'LitePhish_PhishTank':args.phishtank,
    }
    rows=[]
    report={}
    for name,path in datasets.items():
        if not path.is_file():
            report[name]={'status':'missing','path':str(path)}; continue
        df=load(path)
        urlcol='url' if 'url' in df else df.columns[0]
        labelcol='phishing' if 'phishing' in df else 'label'
        assert labelcol in df and set(df[labelcol].dropna().astype(int).unique()).issubset({0,1})
        labels=df[labelcol].astype(int)
        parsed=[parse(u) for u in df[urlcol].fillna('')]
        normalized=pd.Series([r[0] for r in parsed])
        domains=pd.Series([r[1] for r in parsed])
        valid=pd.Series([r[3] for r in parsed])
        has_tail=pd.Series([r[4] for r in parsed])
        overlap_url=normalized.isin(source_urls)
        overlap_domain=domains.isin(source_domains)
        clean=valid & normalized.ne('') & domains.ne('') & ~overlap_url & ~overlap_domain
        duplicate=normalized.duplicated(keep='first')
        conflicting=set(normalized.groupby(normalized).apply(lambda g:labels.loc[g.index].nunique()).loc[lambda x:x>1].index)
        clean &= ~duplicate & ~normalized.isin(conflicting)
        entry={'path':str(path),'sha256':sha(path),'rows':len(df),'columns':list(df.columns),
               'timestamp_columns':[c for c in df if any(s in c.lower() for s in ('date','time','first_seen','collected'))],
               'normalized_url_overlap_with_complete_PhishFusion':int(overlap_url.sum()),
               'registered_domain_overlap_with_complete_PhishFusion':int(overlap_domain.sum()),
               'normalized_duplicate_rows':int(duplicate.sum()),'conflicting_normalized_urls':len(conflicting),
               'eligible_full_url_domain_disjoint_rows':int(clean.sum()),
               'label_provenance':'Inherited dataset annotations; no live page or individual benign verification performed'}
        report[name]=entry
        for label in (0,1):
            mask=labels.eq(label)
            rows.append(dict(dataset=name,label=label,n=int(mask.sum()),
                full_http_url=int((mask&valid).sum()),bare_or_other_scheme=int((mask&~valid).sum()),
                nontrivial_path_or_query=int((mask&has_tail).sum()),
                eligible_full_domain_disjoint=int((mask&clean).sum())))
        # Freeze eligibility only; no model or representation is chosen on outcomes.
        if name!='LitePhish_source_PhishFusion':
            pd.DataFrame({'row_id':range(len(df)),'label':labels,
                'full_http_url':valid,'source_normalized_url_overlap':overlap_url,
                'source_registered_domain_overlap':overlap_domain,
                'normalized_duplicate':duplicate,'eligible_full_domain_disjoint':clean}).to_csv(
                    args.output/(name+'_eligibility.csv.gz'),index=False)
    old=args.reference_root
    artifacts=args.artifacts
    report['frozen_original_artifacts']={p.name:sha(p) for p in artifacts.glob('*.pkl')}
    report['historical_prediction_recomputation']={}
    temporal=old/'experiment_25_temporal_generalization/evaluation/LitePhish_temporal_predictions.csv'
    if temporal.is_file():
        d=load(temporal); pred=d['probability'].ge(.5)
        report['historical_prediction_recomputation']['temporal']={'path':str(temporal),'sha256':sha(temporal),
            'n':len(d),'detected':int(pred.sum()),'recall':float(pred.mean())}
    dns=old/'experiment_29_consumer_benign/frozen_released/LitePhish_consumer_benign_predictions.csv.gz'
    if dns.is_file():
        d=load(dns)
        report['historical_prediction_recomputation']['popular_dns']={'path':str(dns),'sha256':sha(dns),
            'n':len(d),'flagged':int(d['probability'].ge(.5).sum()),'flag_rate':float(d['probability'].ge(.5).mean()),
            'full_http_url_rows':int(d['url'].map(lambda u:parse(u)[3]).sum()),
            'important_boundary':'DNS-derived homepage URLs, not individually verified browser-navigation benign traffic; adding https does not verify benignness'}
    pd.DataFrame(rows).to_csv(args.output/'cohort_profile.csv',index=False)
    (args.output/'dataset_audit.json').write_text(json.dumps(report,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(report['historical_prediction_recomputation'],indent=2))


if __name__=='__main__': main()
