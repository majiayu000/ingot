"""Six paired blocks: accepted Strata full-vocabulary Q4 head vs selected native.
This compares complete requests through each usable entry. HTTP/SSE parsing is
included on the server side; it is not an isolated kernel-efficiency comparison.
"""
from pathlib import Path
import hashlib,json,math,os,subprocess,time
from bench_qwen_extended import execute

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 r=Path(__file__).resolve().parents[1];e=r/'evidence/qwen-mtp-20261013'
 while not (e/'extended-queue-complete.json').exists():
  if (e/'formal-queue-stopped.json').exists():raise SystemExit('Quality gate not passed')
  for n in ['native-c1-jobs.json','long-c1-jobs.json','remaining-checks-resumed.json','stage2-resumed-jobs.json','formal-queue-jobs.json','comparator-queue-jobs.json','glm-mtp-validation-jobs.json','extended-queue-jobs.json']:
   p=e/n
   if p.exists() and any(v['exit'] for v in json.loads(p.read_text())):raise SystemExit('Previous GPU job failed: '+n)
  time.sleep(30)
 out=e/'best-paired';out.mkdir();q=str(Path.home()/'Models/Strata/Qwen3.8-IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf');bin=Path.home()/'.cache/ingot-qwen-online-20261013/release/examples/bench_tokens';selected=json.loads((e/'formal-queue-complete.json').read_text())['selected_config'];head=str(Path.home()/'Models/Strata/MTP/mtp-Qwen3.8-Flash-Next-local-head-Q4_0.gguf');api=r/'tools/bench_qwen_api.py';cases=[(c,n) for n in [1024,4096] for c in ['chinese','code','math']]
 files=[Path(__file__),api,r/'tools/bench_qwen_extended.py',r/'tools/bench_qwen_pair.py',r/'tools/apple_process_memory.py',bin,e/'formal-queue-complete.json',e/'strata-current-frozen/provenance.json',e/'api-quality-strata-q4/comparison.json',e/'api-quality-strata-q4/comparison-vs-native.json',e/'quality-final-complete.json']+[e/f'workloads/{c}-{n}.u32' for c,n in cases]
 freeze={str(p):sha(p) for p in files};quality=json.loads((e/'api-quality-strata-q4/comparison.json').read_text())['summary'];protocol={'pairs':6,'order':'AB BA AB BA AB BA; A Strata current accepted full-vocabulary head-only Q4 MTP3; B selected native','candidate':selected,'sha256':freeze,'primary':'complete requests excluding load, cap256, EOS honored; six equally weighted 1K/4K cases','strata_quality_relative_to_fixed_reference_all2345':quality,'strata_quality_relative_to_native_all2345':json.loads((e/'api-quality-strata-q4/comparison-vs-native.json').read_text())['summary'],'memory':'Native48GiB reservation/Metal/RSS/footprint gates; Strata full-resident standard configuration with measured memory, not an imposed 48GiB cap','entry_difference':'Strata client HTTP/SSE E2E vs native Engine.generate; included overhead is explicit, not a pure-kernel claim','blocks':'Strata one engine per arm with per-case warm+one hot; native one engine per case with warm+one hot. OS file cache is not cleared.','ci':'paired process-block bootstrap10000 draws seed20261013; preserve case correlation inside each pair'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');runs=[];jobs=[]
 env=os.environ.copy()
 for k in ['QWEN_DRAFT','DRAFT_TOKENS','DRAFT_VOCAB','PREFIX_CACHE']:env.pop(k,None)
 if selected['draft']:env['QWEN_DRAFT']=selected['draft'];env['DRAFT_TOKENS']=str(selected['depth'])
 if selected['vocabulary']:env['DRAFT_VOCAB']=selected['vocabulary']
 def job(name,cmd):
  start=time.monotonic()
  with (out/(name+'.stdout')).open('x') as stdout,(out/(name+'.stderr')).open('x') as stderr:code=subprocess.run([str(x) for x in cmd],stdout=stdout,stderr=stderr).returncode
  row={'name':name,'exit':code,'seconds':time.monotonic()-start};jobs.append(row);(e/'best-paired-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');print(json.dumps(row),flush=True)
  if code:raise RuntimeError('paired server block failed')
 for pair in range(6):
  for arm in (['A','B'] if pair%2==0 else ['B','A']):
   if arm=='A':
    directory=out/('pair%d-strata'%pair);job('pair%d-strata'%pair,['/usr/bin/python3',api,'strata',directory,'--depth','3','--draft',head,'--repeats','1','--short'])
    for c,n in cases:
     sample=json.loads((directory/f'{c}-{n}-1.json').read_text());runs.append({'pair':pair,'arm':arm,'category':c,'input':n,'total_s':sample['total_s'],'first_visible_s':sample['first_visible_s'],'output_tokens':sample['output_tokens'],'stop':sample['stop'],'output_sha256':hashlib.sha256(sample['text'].encode()).hexdigest()})
   else:
    for c,n in cases:
     name='pair%d-native-%s-%d'%(pair,c,n);row=execute(out,name,[bin,q,'48',str(n+256),'256','1',e/f'workloads/{c}-{n}.u32'],env,48)
     if row['failure'] is None and (len(row['samples'])!=2 or (row['samples'][0]['tokens'],row['samples'][0]['stop'])!=(row['samples'][1]['tokens'],row['samples'][1]['stop'])):row['failure']='missing metrics/repeated output mismatch'
     (out/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');jobs.append({'name':name,'exit':0 if row['failure'] is None else 98,'failure':row['failure'],'seconds':row['seconds']});(e/'best-paired-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
     if row['failure']:raise RuntimeError(row['failure'])
     sample=row['samples'][1];runs.append({'pair':pair,'arm':arm,'category':c,'input':n,'total_s':sample['total_s'],'first_visible_s':sample['first_visible_s'],'output_tokens':len(sample['tokens']),'stop':sample['stop'],'output_sha256':hashlib.sha256(json.dumps(sample['tokens']).encode()).hexdigest()});print(json.dumps(jobs[-1]),flush=True)
   (out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
   for c,n in cases:
    rows=[v for v in runs if v['arm']==arm and v['category']==c and v['input']==n]
    assert len({(v['output_sha256'],v['stop'],v['output_tokens']) for v in rows})==1,'Repeated greedy outputs changed'
 import numpy as np
 indices=np.random.default_rng(20261013).integers(0,6,size=(10000,6));report={};logs=[]
 for c,n in cases:
  aa=[next(v for v in runs if v['arm']=='A' and v['pair']==i and v['category']==c and v['input']==n) for i in range(6)];bb=[next(v for v in runs if v['arm']=='B' and v['pair']==i and v['category']==c and v['input']==n) for i in range(6)];ratios=np.array([b['total_s']/a['total_s'] for a,b in zip(aa,bb)]);logs.append(np.log(ratios));boot=np.exp(np.log(ratios)[indices].mean(axis=1));row={'paired_B_over_A':ratios.tolist(),'geometric_ratio':float(np.exp(np.log(ratios).mean())),'ci95':np.quantile(boot,[.025,.975]).tolist()}
  for arm,values in [('A',aa),('B',bb)]:
   times=np.array([v['total_s'] for v in values]);row[arm]={'median_total_s':float(np.median(times)),'range_total_s':[float(times.min()),float(times.max())],'sample_variance_s2':float(np.var(times,ddof=1)),'output_lengths':[v['output_tokens'] for v in values],'first_visible_s':[v['first_visible_s'] for v in values]}
  report[f'{c}-{n}']=row
 mean_logs=np.array(logs).mean(axis=0);boot=np.exp(mean_logs[indices].mean(axis=1));report['equal_weight_total']={'geometric_ratio':float(np.exp(mean_logs.mean())),'ci95':np.quantile(boot,[.025,.975]).tolist(),'conditional_10pct_speed_target':bool(np.exp(mean_logs.mean())<=.9 and np.quantile(boot,.975)<1),'quality_review_required':True}
 assert all(sha(p)==v for p,v in freeze.items()),'paired benchmark inputs changed'
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(e/'best-paired-complete.json').write_text(json.dumps({'completed':True,'summary':str(out/'summary.json'),'not_a_pure_kernel_ratio':True,'not_an_automatic_adoption':True},indent=2)+'\n')
if __name__=='__main__':main()
