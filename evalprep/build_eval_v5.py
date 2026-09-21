# -*- coding: utf-8 -*-
import json, sys, re, unicodedata
sys.path.insert(0, __file__.rsplit('/',1)[0])
from spec_a import SPEC_A
from spec_b import SPEC_B
from spec_c import SPEC_C

CORPUS = sys.argv[1]
OUT    = sys.argv[2]
SPEC   = SPEC_A + SPEC_B + SPEC_C

def eligible(d):
    u = (d.get('url') or '').strip()
    if not u: return False
    t = d.get('text') or ''
    if len(t) < 150: return False
    if str(d.get('low_content')) == 'True': return False
    if str(d.get('boilerplate')) == 'True': return False
    lang = (d.get('lang') or '').strip().lower().split('-')[0]
    if lang and lang not in ('it','en'): return False
    return True

answerable = [q for q in SPEC if q.get('support')]
supports   = [(q['id'], q['support'].lower()) for q in answerable]
hits       = {q['id']: [] for q in answerable}
targets    = {q['id']: q['url'] for q in answerable}
tmeta      = {}

for line in open(CORPUS, encoding='utf-8'):
    try: d = json.loads(line)
    except Exception: continue
    if not eligible(d): continue
    url = d['url']; text = d.get('text') or ''
    low = text.lower()
    for qid, s in supports:
        if s in low:
            hits[qid].append((url, d))
    for qid, turl in targets.items():
        if url == turl:
            tmeta[qid] = d

def snippet(text, support, width=320):
    i = text.lower().find(support.lower())
    if i < 0: return None
    a = max(0, i - width//3); b = min(len(text), i + len(support) + width)
    s = text[a:b].replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return ('...' if a else '') + s + ('...' if b < len(text) else '')

out = []
report = []
for q in SPEC:
    rec = {
        'id': q['id'],
        'question': q['q'],
        'question_lang': q['lang'],
        'stratum': q['stratum'],
        'scoring': q.get('scoring', 'retrieval_and_answer'),
        'gold_answer': q['gold'],
        'objective': q['obj'],
    }
    if q.get('flags'): rec['flags'] = q['flags']
    if q.get('expect'): rec['expected_behaviour'] = q['expect']

    if not q.get('support'):
        rec['target_url'] = None
        rec['acceptable_urls'] = []
        if q.get('requires_poisoned_doc'):
            rec['requires_poisoned_doc'] = True
            rec['poison_doc'] = q['poison']
        out.append(rec); report.append((q['id'], 'no-retrieval', 0, 'ok')); continue

    qid = q['id']
    hs = hits[qid]
    tm = tmeta.get(qid)
    status = 'ok'
    if tm is None:
        status = 'TARGET-MISSING-OR-INELIGIBLE'
    urls = [u for u, _ in hs]
    if q.get('url_must_match'):
        import re as _re
        _f = _re.compile(q['url_must_match'], _re.I)
        urls = [u for u in urls if _f.search(u)]
    # target first, then the rest, deduped, capped
    cap = q.get('max_urls', 8)
    extra = [u for u in q.get('also', []) if u not in urls]
    urls = urls + [u for u in q.get('also', []) if u in {x for x, _ in hs} or True and u not in urls]
    ordered = ([q['url']] if q['url'] in urls else []) + [u for u in urls if u != q['url']]
    seen = set(); acc = []
    for u in ordered:
        if u in seen: continue
        seen.add(u); acc.append(u)
    truncated = len(acc) > cap
    acc = acc[:cap]
    if q['url'] not in urls:
        status = 'SUPPORT-NOT-IN-TARGET' if status == 'ok' else status + '+SUPPORT-NOT-IN-TARGET'

    rec['target_url'] = q['url']
    if q.get('distractor_url'): rec['distractor_url'] = q['distractor_url']
    rec['acceptable_urls'] = acc
    rec['n_supporting_docs_in_corpus'] = len(urls)
    if truncated: rec['acceptable_urls_truncated'] = True
    rec['evidence_query'] = q['support']
    if tm is not None:
        rec['evidence'] = snippet(tm.get('text',''), q['support'])
        rec['title'] = tm.get('title')
        rec['corpus_lang'] = (tm.get('lang') or '') or None
        rec['corpus_year'] = tm.get('effective_year')
        rec['corpus_academic_year'] = tm.get('academic_year')
        rec['corpus_chars'] = len(tm.get('text') or '')
        rec['extractor'] = tm.get('extractor')
        rec['source_note'] = tm.get('note')
    out.append(rec)
    report.append((qid, status, len(urls), acc[0] if acc else '-'))

out.sort(key=lambda r: r['id'])
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f'wrote {len(out)} questions -> {OUT}')
print()
print(f"{'id':>3}  {'status':<34} {'nsup':>5}  target/first-acceptable")
bad = 0
for qid, st, n, u in sorted(report):
    if st not in ('ok','no-retrieval'): bad += 1
    print(f'{qid:>3}  {st:<34} {n:>5}  {u[:95]}')
print()
print('problems:', bad)
