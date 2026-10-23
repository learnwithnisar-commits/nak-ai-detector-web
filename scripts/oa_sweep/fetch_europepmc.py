"""Europe PMC: OA abstracts (diverse queries) + OA full texts -> oa_corpus/europepmc/."""
import sys, os, time, json, urllib.request, urllib.parse, re
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(__file__))
from common import CorpusWriter, load_seed_seen

ABS_QUERIES = [
    'diabetes mellitus', 'cancer immunotherapy', 'CRISPR gene editing',
    'gut microbiome', 'alzheimer disease', 'vaccine efficacy',
    'antibiotic resistance', 'malaria', 'tuberculosis', 'cardiovascular risk',
]
FT_QUERIES = [
    'stroke rehabilitation', 'asthma', 'depression treatment', 'obesity',
    'pneumonia', 'hepatitis', 'dengue', 'nutrition', 'surgery outcomes',
    'radiology', 'genomics', 'sepsis',
]
FT_PER_QUERY = 250

def get(url, timeout=45):
    req = urllib.request.Request(url, headers={'User-Agent': 'NAK-AI-Detector/1.0 (mailto:scout@example.com)'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def search(query, page_size=1000, page=1, cursor=None):
    url = ('https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='
           + urllib.parse.quote(query) + f'&format=json&pageSize={page_size}&resultType=core')
    if cursor is not None:
        url += '&cursorMark=' + urllib.parse.quote(cursor, safe='')
    else:
        url += f'&page={page}'
    return json.loads(get(url))

def body_text(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return ''
    # strip namespaces
    for el in root.iter():
        if '}' in el.tag: el.tag = el.tag.split('}', 1)[1]
    parts = []
    for body in root.iter('body'):
        for p in body.iter('p'):
            t = ''.join(p.itertext()).strip()
            if t: parts.append(t)
    return '\n\n'.join(parts)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fulltext-only', action='store_true')
    args = ap.parse_args()
    seed = load_seed_seen()
    w = CorpusWriter('europepmc', seed_seen=seed)
    if not args.fulltext_only:
        # 1) abstracts
        for qi, q in enumerate(ABS_QUERIES):
            try:
                data = search(q)
            except Exception as e:
                print(f'abs [{qi}] ERR {str(e)[:60]}', flush=True); continue
            n = 0
            for r in data.get('resultList', {}).get('result', []):
                abst = (r.get('abstractText') or '').strip()
                if not abst: continue
                pmcid = r.get('pmcid') or r.get('id')
                if w.add(pmcid, r.get('title') or '', abst,
                         doi=r.get('doi'), date=r.get('firstPublicationDate'),
                         license='open-access' if r.get('isOpenAccess') == 'Y' else None,
                         extra={'pmid': r.get('pmid'), 'kind': 'abstract'}):
                    n += 1
            print(f'abs [{qi+1}/{len(ABS_QUERIES)}] {q}: new {n} (total {w.n_new})', flush=True)
            time.sleep(1)
    # 2) full texts -> SEPARATE writer so merge can prefer fulltext over abstract
    print('--- full texts ---', flush=True)
    wf = CorpusWriter('europepmc_ft', seed_seen=seed)
    ft_done = 0
    seen_pmc = set()
    for fqi, fq in enumerate(FT_QUERIES):
        cursor = '*'
        q_done = 0
        while q_done < FT_PER_QUERY:
            try:
                data = search(f'({fq}) AND OPEN_ACCESS:Y AND IN_PMC:Y', page_size=500, cursor=cursor)
            except Exception as e:
                print(f'ft [{fq}] search ERR', str(e)[:60], flush=True); break
            results = data.get('resultList', {}).get('result', [])
            if not results: break
            for r in results:
                pmcid = r.get('pmcid')
                if not pmcid or pmcid in seen_pmc: continue
                seen_pmc.add(pmcid)
                try:
                    xml = get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML', timeout=60)
                except Exception:
                    continue
                txt = body_text(xml)
                if len(txt) < 2000 or len(txt) > 500000:
                    continue
                if wf.add(pmcid, r.get('title') or '', txt, doi=r.get('doi'),
                          date=r.get('firstPublicationDate'), license='open-access',
                          extra={'pmid': r.get('pmid'), 'kind': 'fulltext'}):
                    ft_done += 1
                    q_done += 1
                if ft_done % 250 == 0:
                    print(f'  fulltext: {ft_done} (q {fqi+1}/{len(FT_QUERIES)})', flush=True)
                time.sleep(0.35)
            cursor = data.get('nextCursorMark')
            if not cursor: break
        print(f'ft [{fqi+1}/{len(FT_QUERIES)}] {fq}: total {ft_done}', flush=True)
    wf.close()
    w.close()

if __name__ == '__main__':
    main()
