"""DOAJ articles API: OA journal articles -> oa_corpus/doaj/."""
import sys, os, time, json, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(__file__))
from common import CorpusWriter, load_seed_seen

SUBJECTS = ['medicine', 'biology', 'physics', 'chemistry', 'computer science',
            'engineering', 'mathematics', 'psychology', 'economics', 'environmental science']
PER_PAGE, MAX_PAGES = 100, 10

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'NAK-AI-Detector/1.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def bib_doi(b):
    for ident in b.get('identifier', []) or []:
        if ident.get('type') == 'doi':
            return ident.get('id')
    for link in b.get('link', []) or []:
        u = link.get('url', '')
        if 'doi.org' in u: return u
    return None

def main():
    w = CorpusWriter('doaj', seed_seen=load_seed_seen())
    for si, subj in enumerate(SUBJECTS):
        for pg in range(1, MAX_PAGES + 1):
            url = ('https://doaj.org/api/search/articles/'
                   + urllib.parse.quote(subj) + f'?pageSize={PER_PAGE}&page={pg}')
            try:
                data = get(url)
            except Exception as e:
                print(f'[{subj}] pg{pg} ERR {str(e)[:60]}', flush=True); break
            results = data.get('results', [])
            if not results: break
            n = 0
            for r in results:
                b = r.get('bibjson') or {}
                title = (b.get('title') or '').strip()
                abst = (b.get('abstract') or '').strip()
                if not title or not abst: continue
                lic = None
                for L in b.get('license') or []:
                    if L.get('open_access'): lic = L.get('title'); break
                j = b.get('journal') or {}
                if w.add(r.get('id', ''), title, abst, doi=bib_doi(b),
                         date=str(b.get('year') or ''),
                         license=lic or 'open-access',
                         extra={'journal': j.get('title'),
                                'subjects': b.get('subject')}):
                    n += 1
            print(f'[{si+1}/{len(SUBJECTS)}] {subj} pg{pg}: new {n} (total {w.n_new})', flush=True)
            time.sleep(1.2)
    w.close()

if __name__ == '__main__':
    main()
