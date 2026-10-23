"""OpenAlex OA works -> oa_corpus/openalex/. Fast via cursor pagination (polite pool)."""
import sys, os, time, json, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(__file__))
from common import CorpusWriter, load_seed_seen, CORPUS

TARGET = 30000
PER = 200
MAILTO = 'scout@example.com'

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': f'NAK-AI-Detector/1.0 (mailto:{MAILTO})'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def recon_abstract(inv):
    if not inv: return None
    pos = {}
    for w, idxs in inv.items():
        for i in idxs: pos[i] = w
    return ' '.join(pos[i] for i in sorted(pos))

def main():
    w = CorpusWriter('openalex', seed_seen=load_seed_seen())
    s2_dois = []
    cursor = '*'
    filt = 'is_oa:true,from_publication_date:2015-01-01'
    sel = 'id,doi,title,publication_date,abstract_inverted_index,open_access,best_oa_location'
    n_noabs = 0
    while w.n_new < TARGET:
        url = ('https://api.openalex.org/works?filter=' + urllib.parse.quote(filt)
               + f'&per-page={PER}&cursor={urllib.parse.quote(cursor)}'
               + f'&select={sel}&mailto={MAILTO}')
        try:
            data = fetch(url)
        except Exception as e:
            print('fetch err', str(e)[:80], flush=True); time.sleep(5); continue
        results = data.get('results', [])
        if not results: break
        for r in results:
            title = r.get('title') or ''
            if not title or title.lower() == 'null': continue
            abs_txt = recon_abstract(r.get('abstract_inverted_index'))
            oa = r.get('open_access') or {}
            bol = r.get('best_oa_location') or {}
            lic = oa.get('license') or bol.get('license')
            if abs_txt:
                w.add(r['id'].split('/')[-1], title, abs_txt, doi=r.get('doi'),
                      date=r.get('publication_date'), license=lic,
                      extra={'oa_url': bol.get('pdf_url') or bol.get('landing_page_url'),
                             'oa_status': oa.get('oa_status')})
            else:
                n_noabs += 1
                d = (r.get('doi') or '').strip()
                if d and len(s2_dois) < 15000 and d not in w.seen_doi:
                    s2_dois.append(d)
        cursor = data.get('meta', {}).get('next_cursor')
        if not cursor: break
        if w.n_new % 2000 < PER:
            print(f'  openalex: {w.n_new} docs, {n_noabs} no-abstract', flush=True)
        time.sleep(0.15)
    w.close()
    with open(os.path.join(CORPUS, 's2_doi_list.txt'), 'w') as f:
        f.write('\n'.join(s2_dois))
    print(f's2 doi list: {len(s2_dois)}', flush=True)

if __name__ == '__main__':
    main()
