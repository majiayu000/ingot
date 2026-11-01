"""Frozen natural-context coverage scorer; no population equivalence claim."""
import decimal, hashlib, json, sys
from pathlib import Path
from quality_qwen import normalize_zh, char_lcs_f1, math_answer, paired_bound

manifest, reference, autoregressive, speculative, output = map(Path, sys.argv[1:])
m=json.loads(manifest.read_text())
def load(p):return {(r['category'],r['id']):r for r in map(json.loads,p.read_text().splitlines())}
arms=[load(p) for p in (reference,autoregressive,speculative)]
expected={(t['category'],t['id']) for t in m['tasks']}
assert all(set(x)==expected for x in arms)
rows=[]
for task in m['tasks']:
    key=task['category'],task['id'];scores=[]
    for arm in arms:
        generation=arm[key];text=generation['text'];cat=key[0]
        if cat=='math':correct=math_answer(text)[0] in [decimal.Decimal(a) for a in task['answers']]
        elif cat=='code':correct=text.strip().strip('`').strip() in task['answers']
        else:correct=normalize_zh(text) in [normalize_zh(a) for a in task['answers']]
        scores.append({'correct':correct,'truncated':generation['stop']=='Length',
                       'stop':generation['stop'],'text':text,'output_tokens':len(generation['tokens']),
                       'char_lcs_f1':max(char_lcs_f1(text,a) for a in task['answers'])})
    a,b=arms[1][key],arms[2][key]
    exact=a['tokens']==b['tokens'] and a['stop']==b['stop']
    first=next((i for i,(x,y) in enumerate(zip(a['tokens'],b['tokens'])) if x!=y),None)
    if first is None and len(a['tokens'])!=len(b['tokens']):first=min(len(a['tokens']),len(b['tokens']))
    rows.append({'category':key[0],'id':key[1],'context':task['nominal_context'],
                 'reference':scores[0],'ar':scores[1],'mtp':scores[2],
                 'mtp_exact':exact,'first_token_divergence':first})
summary={}
for name in ['zh','en','code','math','2048','4096','8192','total']:
    subset=[x for x in rows if name=='total' or x['category']==name or str(x['context'])==name]
    d=[int(x['mtp']['correct'])-int(x['reference']['correct']) for x in subset]
    interval=paired_bound(d);interval.pop('passed')
    summary[name]={**interval,'mtp_exact':all(x['mtp_exact'] for x in subset),
                   'correct':{arm:sum(x[arm]['correct'] for x in subset) for arm in ['reference','ar','mtp']},
                   'truncated':{arm:sum(x[arm]['truncated'] for x in subset) for arm in ['reference','ar','mtp']}}
result={'coverage_only':True,'notes':['24 examples do not establish population quality equivalence.','Strict answer extraction is primary; auxiliary character F1 is descriptive.','The paired interval uses the pre-existing conservative bound; no 2pp quality gate is applied to this small coverage set.'],
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'summary':summary,'rows':rows}
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
if not summary['total']['mtp_exact']:raise SystemExit('MTP / AR output divergence requires diagnosis')
