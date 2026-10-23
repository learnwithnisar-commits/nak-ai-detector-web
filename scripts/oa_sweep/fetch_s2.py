"""Semantic Scholar: batch-resolve DOIs (from OpenAlex no-abstract list) -> oa_corpus/semanticscholar/."""
import sys, os, time, json, urllib.request, re
sys.path.insert(0, os.path.dirname(__file__))
from common import CorpusWriter, load_seed_seen, CORPUS

BATCH = 500  # ids per request
FIELDS = 'title,abstract,year,publicationDate,openAccessPdf,externalIds,venue'

def batch_fetch(ids):
    payload = json.dumps({'ids': ['DOI:' + d for d in ids]}).encode()
    req = urllib.request.Request(
        'https://api.semanticscholar.org/graph/v1/paper/batch?fields=' + FIELDS,
        data=payload, headers={'Content-Type': 'application/json'},
        method='POST')
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def main():
    lp = os.path.join(CORPUS, 's2_doi_list.txt')
    if not os.path.exists(lp):
        print('no s2_doi_list.txt yet', flush=True); return
    dois = [l.strip() for l in open(lp) if l.strip()]
    # normalize to bare DOIs (list may contain https://doi.org/... URLs)
    bare = []
    for d in dois:
        d = re.sub(r'^https?://(dx\.)?doi\.org/', '', d).strip().lower()
        if d.startswith('10.'): bare.append(d)
    dois = bare
    print(f's2: {len(dois)} dois to resolve', flush=True)
    w = CorpusWriter('semanticscholar', seed_seen=load_seed_seen())
    for i in range(0, len(dois), BATCH):
        chunk = dois[i:i+BATCH]
        papers = None
        for attempt in range(4):
            try:
                papers = batch_fetch(chunk)
                break
            except Exception as e:
                wait = 70 * (attempt + 1)
                print(f'batch {i//BATCH} attempt {attempt} ERR {str(e)[:60]}; wait {wait}s', flush=True)
                time.sleep(wait)
        if papers is None:
            print(f'batch {i//BATCH} FAILED after retries, skipping {len(chunk)} dois', flush=True)
            continue
        for d, p in zip(chunk, papers):
            if not p or not p.get('abstract'): continue
            ext = p.get('externalIds') or {}
            oap = p.get('openAccessPdf') or {}
            w.add('s2_' + str(abs(hash(d)) % 10**10), p.get('title') or '', p['abstract'],
                  doi=d, date=p.get('publicationDate') or str(p.get('year') or ''),
                  license='open-access' if oap.get('url') else None,
                  extra={'oa_url': oap.get('url'), 'venue': p.get('venue'),
                         's2_doi': ext.get('DOI')})
        print(f'  s2: {w.n_new} docs after {i+BATCH} dois', flush=True)
        time.sleep(6)  # ~100 req / 5 min unauthenticated
    w.close()

if __name__ == '__main__':
    main()
