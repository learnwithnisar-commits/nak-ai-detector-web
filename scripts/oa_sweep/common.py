"""Shared utilities for the OA corpus sweep (dual use: AI training + plagiarism fingerprints)."""
import json, os, re, hashlib, zlib
import numpy as np

CORPUS = os.path.expanduser('~/workspace/ai-detector/data/oa_corpus')
os.makedirs(CORPUS, exist_ok=True)

def clean_text(t):
    t = re.sub(r'\s+', ' ', t or '').strip()
    return t

def norm_title(t):
    return re.sub(r'[^a-z0-9]', '', (t or '').lower())

def title_hash(t):
    return hashlib.sha1(norm_title(t).encode()).hexdigest()[:16]

def norm_doi(d):
    d = (d or '').strip().lower()
    d = re.sub(r'^https?://(dx\.)?doi\.org/', '', d)
    return d or None

def safe_id(s):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s)[:120]

# ---------- MinHash (numpy-vectorized splitmix64, 64 perms, k=5 word shingles) ----------
_P = 64
_SEEDS = (np.arange(_P, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(1))

def minhash_sig(text, k=5, p=_P):
    words = text.lower().split()
    if len(words) < k:
        return None, 0
    shingles = {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}
    xs = np.array([zlib.crc32(s.encode('utf-8')) for s in shingles], dtype=np.uint64)
    with np.errstate(over='ignore'):
        z = xs[None, :] + _SEEDS[:p, None]
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        sig = z.min(axis=1)
    return [int(v) for v in sig], len(shingles)

class CorpusWriter:
    """Per-source writer: text files + doc records + fingerprints, with in-source dedup."""
    def __init__(self, source, seed_seen=None):
        self.source = source
        self.d = os.path.join(CORPUS, source)
        os.makedirs(self.d, exist_ok=True)
        self.docs_path = os.path.join(CORPUS, f'{source}_docs.jsonl')
        self.fp_path = os.path.join(CORPUS, f'fp_{source}.jsonl')
        self.seen_doi = set()
        self.seen_th = set()
        if seed_seen:
            self.seen_doi.update(seed_seen.get('dois', ()))
            self.seen_th.update(seed_seen.get('titlehashes', ()))
        self.n_new = 0
        self.n_dup = 0
        self._docs = open(self.docs_path, 'a', encoding='utf-8')
        self._fp = open(self.fp_path, 'a', encoding='utf-8')

    def add(self, doc_id, title, text, doi=None, date=None, license=None, extra=None):
        text = clean_text(text)
        nw = len(text.split())
        if nw < 30:  # too short to be useful
            return False
        doi = norm_doi(doi)
        th = title_hash(title)
        if (doi and doi in self.seen_doi) or th in self.seen_th:
            self.n_dup += 1
            return False
        if doi: self.seen_doi.add(doi)
        self.seen_th.add(th)
        sid = safe_id(doc_id)
        tp = os.path.join(self.d, sid + '.txt')
        # avoid clobbering on rerun
        if os.path.exists(tp):
            sid = sid + '_' + th[:6]
            tp = os.path.join(self.d, sid + '.txt')
        with open(tp, 'w', encoding='utf-8') as f:
            f.write(text)
        sig, n_sh = minhash_sig(text)
        rec = {'doc_id': f'{self.source}:{sid}', 'source': self.source, 'doi': doi,
               'title': title, 'date': date, 'license': license,
               'text_path': os.path.relpath(tp, CORPUS), 'n_words': nw,
               'train_eligible': nw >= 50, 'title_hash': th}
        if extra: rec.update(extra)
        self._docs.write(json.dumps(rec, ensure_ascii=False) + '\n')
        if sig:
            self._fp.write(json.dumps({'doc_id': rec['doc_id'], 'sig': sig,
                                       'n_shingles': n_sh, 'k': 5, 'p': _P}) + '\n')
        self.n_new += 1
        return True

    def close(self):
        self._docs.close(); self._fp.close()
        print(f'[{self.source}] new={self.n_new} dup/skipped={self.n_dup}', flush=True)

def load_seed_seen():
    """Seed dedup from the already-used real_human training data."""
    dois, ths = set(), set()
    base = os.path.expanduser('~/workspace/ai-detector/data/real_human')
    for fn in os.listdir(base):
        if not fn.endswith('.jsonl'): continue
        try:
            with open(os.path.join(base, fn), encoding='utf-8') as f:
                for line in f:
                    try: r = json.loads(line)
                    except: continue
                    t = r.get('title') or ''
                    if t: ths.add(title_hash(t))
                    d = norm_doi(r.get('doi'))
                    if d: dois.add(d)
        except Exception:
            pass
    print(f'seed seen: {len(dois)} dois, {len(ths)} titlehashes', flush=True)
    return {'dois': dois, 'titlehashes': ths}
