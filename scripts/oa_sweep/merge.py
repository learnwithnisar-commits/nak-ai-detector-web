"""Merge per-source sweeps: global dedup (fulltext first, then source priority),
global manifest.jsonl, merged fingerprints, stats."""
import json, os

import sys
sys.path.insert(0, os.path.dirname(__file__))
from common import CORPUS, norm_doi, title_hash

SRC_PRIORITY = {'europepmc_ft': 0, 'europepmc': 1, 'arxiv': 2, 'openalex': 3,
                'semanticscholar': 4, 'doaj': 5}

def main():
    recs = []
    for src in SRC_PRIORITY:
        dp = os.path.join(CORPUS, f'{src}_docs.jsonl')
        if not os.path.exists(dp): continue
        with open(dp, encoding='utf-8') as f:
            for line in f:
                try: recs.append(json.loads(line))
                except: pass
    print(f'raw records: {len(recs)}', flush=True)

    def rank(r):
        ft = 0 if r.get('kind') == 'fulltext' else 1
        return (ft, SRC_PRIORITY.get(r.get('source'), 9))
    recs.sort(key=rank)

    seen_doi, seen_th = set(), set()
    kept, dropped = [], []
    for r in recs:
        doi = norm_doi(r.get('doi'))
        th = r.get('title_hash') or title_hash(r.get('title'))
        if (doi and doi in seen_doi) or th in seen_th:
            dropped.append(r); continue
        if doi: seen_doi.add(doi)
        seen_th.add(th)
        kept.append(r)

    # remove dropped duplicate text files to keep corpus clean
    for r in dropped:
        tp = os.path.join(CORPUS, r.get('text_path', ''))
        try:
            if tp.startswith(CORPUS) and os.path.exists(tp): os.remove(tp)
        except Exception: pass

    # global manifest
    mp = os.path.join(CORPUS, 'manifest.jsonl')
    with open(mp, 'w', encoding='utf-8') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # merged fingerprints (only kept docs) — explicit source list to avoid stray files
    keep_ids = {r['doc_id'] for r in kept}
    fp_out = os.path.join(CORPUS, 'fingerprints', 'minhash.jsonl')
    os.makedirs(os.path.dirname(fp_out), exist_ok=True)
    n_fp = 0
    seen_fp = set()
    with open(fp_out, 'w', encoding='utf-8') as out:
        for src in SRC_PRIORITY:
            fp = os.path.join(CORPUS, f'fp_{src}.jsonl')
            if not os.path.exists(fp): continue
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    try: j = json.loads(line)
                    except: continue
                    did = j.get('doc_id')
                    if did in keep_ids and did not in seen_fp and 'sig' in j:
                        out.write(line); n_fp += 1; seen_fp.add(did)

    # stats
    from collections import Counter
    by_src = Counter(r['source'] for r in kept)
    by_kind = Counter(r.get('kind', 'abstract') for r in kept)
    train_ok = sum(1 for r in kept if r.get('train_eligible'))
    disk = sum(os.path.getsize(os.path.join(CORPUS, r['text_path']))
               for r in kept if os.path.exists(os.path.join(CORPUS, r['text_path'])))
    words = sum(r.get('n_words', 0) for r in kept)
    stats = {
        'total_docs': len(kept), 'dropped_dup': len(dropped),
        'by_source': dict(by_src), 'by_kind': dict(by_kind),
        'train_eligible': train_ok, 'fingerprint_only': len(kept) - train_ok,
        'fingerprints': n_fp,
        'total_words': words, 'disk_mb': round(disk / 1e6, 1),
    }
    with open(os.path.join(CORPUS, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)

if __name__ == '__main__':
    main()
