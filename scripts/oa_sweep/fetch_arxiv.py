"""arXiv expansion: more pages per category + new categories -> oa_corpus/arxiv/."""
import sys, os, time, json, subprocess, urllib.parse, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(__file__))
from common import CorpusWriter, load_seed_seen

NS = {'a': 'http://www.w3.org/2005/Atom'}
CATEGORIES = [
    ['cs.CL', 'cs.AI', 'cs.LG'], ['stat.ML', 'math.ST', 'math.PR'],
    ['astro-ph', 'quant-ph'], ['q-bio.QM', 'q-bio.GN'], ['econ.EM', 'cs.CY'],
    ['cs.CV', 'cs.RO'], ['q-fin.PR', 'q-fin.EC'], ['stat.AP', 'math.OC'],
    ['physics.bio-ph', 'physics.soc-ph'], ['cs.HC', 'cs.SE'],
    ['math.NA', 'cs.NA'], ['cond-mat.stat-mech', 'cond-mat.dis-nn'],
]
PER_PAGE, PAGES = 500, 4
DELAY = 3.2

def fetch(url):
    # curl is more reliable through the egress proxy than urllib here;
    # proxy is flaky for arXiv -> retry with backoff, validate feed XML
    last_err = ''
    for attempt in range(4):
        try:
            out = subprocess.run(['curl', '-sSL', '--max-time', '50', '-A', 'NAK-AI-Detector/1.0', url],
                                 capture_output=True, timeout=60)
            data = out.stdout
            if out.returncode == 0 and b'<feed' in data and b'</feed>' in data:
                return data
            last_err = (out.stderr.decode()[:80] or 'bad xml')
        except Exception as e:
            last_err = str(e)[:80]
        time.sleep(3 * (attempt + 1))
    raise RuntimeError('arxiv fetch failed: ' + last_err)

def main():
    w = CorpusWriter('arxiv', seed_seen=load_seed_seen())
    for qi, cats in enumerate(CATEGORIES):
        q = ' OR '.join('cat:%s' % c for c in cats)
        qlabel = '+'.join(cats)
        # first 6 groups were fetched yesterday (newest ~500); skip their page 0
        pgs = range(1, PAGES) if qi < 6 else range(PAGES)
        for pg in pgs:
            start = pg * PER_PAGE
            params = {'search_query': q, 'start': start, 'max_results': PER_PAGE,
                      'sortBy': 'submittedDate', 'sortOrder': 'descending'}
            url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode(params)
            try:
                xml = fetch(url)
                root = ET.fromstring(xml)
            except Exception as e:
                print(f'[{qi}] pg{pg} FAIL {str(e)[:60]}', flush=True); time.sleep(DELAY); continue
            n = 0
            for e in root.findall('a:entry', NS):
                def _txt(tag):
                    el = e.find(tag, NS)
                    return (el.text or '').strip() if el is not None and el.text else ''
                aid = _txt('a:id').rsplit('/abs/', 1)[-1].replace('/', '_')
                title = _txt('a:title')
                abst = _txt('a:summary')
                pub = _txt('a:published')[:10]
                if not aid or not abst:
                    continue
                if w.add(aid, title, abst, date=pub,
                         extra={'arxiv_id': aid}):
                    n += 1
            print(f'[{qi+1}/{len(CATEGORIES)}] {qlabel} pg{pg}: new {n} (total {w.n_new})', flush=True)
            time.sleep(DELAY)
    w.close()

if __name__ == '__main__':
    main()
