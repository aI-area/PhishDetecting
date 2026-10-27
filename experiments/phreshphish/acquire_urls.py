"""Retrieve public URL metadata only; never visit dataset URLs or run a model."""
import argparse,csv,datetime,hashlib,json,time,urllib.request
from pathlib import Path
from collections import Counter

REPO='phreshphish/phreshphish'
COMMIT='eca603e0ecd4f698f8d4d226c088659e72681239'
def get(url):
    with urllib.request.urlopen(url,timeout=60) as response:return response.read()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    metadata=json.loads(Path(__file__).with_name('shards.json').read_text())
    paths=metadata['shards']
    card=get('https://huggingface.co/datasets/'+REPO+'/raw/'+COMMIT+'/README.md')
    (a.output/'DATASET_CARD.md').write_bytes(card)
    assert len(paths)==21
    columns=['source_id','url','source_label','collection_date','language','source_shard','source_shard_row','url_sha256']
    plan=dict(repository='https://huggingface.co/datasets/'+REPO,dataset_version=COMMIT,split='test',shards=paths,columns=columns,
              selection='All released test rows; extract benign-labelled rows without model-based selection.',
              purpose='URL-only external evaluation.',
              license='CC BY 4.0; anti-phishing research use',paper='https://arxiv.org/abs/2507.10854',
              limitations=['A mixture of anonymized browsing telemetry and brand-related search results, not a probability sample of all browsing.',
                           'Human-assisted cleaning does not imply individual human verification of every row.',
                           'No independence from LitePhish training data is asserted before a separate overlap assessment.',
                           'No test result or deployment effectiveness is inferred from acquisition.'],
              live_dataset_urls_visited=False,models_run=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (a.output/'ACQUISITION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    import duckdb
    con=duckdb.connect();con.execute('LOAD httpfs');con.execute("SET memory_limit='512MB'");con.execute('SET threads=2')
    counts=Counter();dates=[];completed=[];start=time.monotonic()
    partial=a.output/'test_url_metadata.csv.partial';benign_partial=a.output/'benign_urls.csv.partial'
    with partial.open('x',newline='',encoding='utf-8') as full,benign_partial.open('x',newline='',encoding='utf-8') as benign:
        w=csv.DictWriter(full,fieldnames=columns);w.writeheader();bw=csv.DictWriter(benign,fieldnames=columns);bw.writeheader()
        for shard in paths:
            url='https://huggingface.co/datasets/'+REPO+'/resolve/'+COMMIT+'/'+shard
            rows=con.execute('SELECT sha256,url,label,CAST(date AS VARCHAR),lang FROM read_parquet(?)',[url]).fetchall()
            for i,(source_id,text,label,date,language) in enumerate(rows):
                assert label in ['benign','phish'] and isinstance(text,str)
                row=dict(source_id=source_id,url=text,source_label=label,collection_date=date,language=language,source_shard=shard,source_shard_row=i,url_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
                w.writerow(row)
                if label=='benign':bw.writerow(row)
                counts[label]+=1
                if date:dates.append(date)
            full.flush();benign.flush();completed.append(dict(shard=shard,rows=len(rows)))
            (a.output/'progress.json').write_text(json.dumps(dict(completed=completed,counts=dict(counts)),indent=2)+'\n')
            print(json.dumps(dict(shards=len(completed),of=len(paths),counts=dict(counts))),flush=True)
    partial.rename(a.output/'test_url_metadata.csv');benign_partial.rename(a.output/'benign_urls.csv')
    assert counts=={'benign':91260,'phish':76800},'Released counts differ from current dataset card'
    result=dict(status='URL_METADATA_ACQUIRED',dataset_version=COMMIT,counts=dict(counts),total=sum(counts.values()),date_min=min(dates),date_max=max(dates),elapsed_s=time.monotonic()-start,duckdb_version=duckdb.__version__,live_dataset_urls_visited=False,url_metadata_only=True)
    result['files_sha256']={p.name:digest(p) for p in a.output.iterdir() if p.is_file()}
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
