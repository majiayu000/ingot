"""Frozen budget calibration and four-round prefix measurements; owned GPU lane only."""
from pathlib import Path
import argparse,hashlib,json,math,os,re,signal,subprocess,time
from apple_process_memory import MemoryMonitor
from bench_qwen_pair import environment

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def execute(out,name,cmd,env,budget,timeout=1800):
 before=environment();start=time.monotonic();failure=None
 with (out/(name+'.stdout')).open('x') as stdout,(out/(name+'.stderr')).open('x') as stderr:
  p=subprocess.Popen(['/usr/bin/time','-l',*[str(x) for x in cmd]],env=env,stdout=stdout,stderr=stderr,start_new_session=True);monitor=MemoryMonitor(p.pid,True)
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   failure='timeout';os.killpg(p.pid,signal.SIGTERM)
   try:code=p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
  finally:memory=monitor.close()
 rows=[json.loads(l) for l in (out/(name+'.stdout')).read_text().splitlines() if l.startswith('{')]
 match=re.search(r'(\d+)\s+maximum resident set size',(out/(name+'.stderr')).read_text());rss=int(match[1]) if match else None
 if code:failure=failure or 'process failure'
 elif memory['errors'] or not memory['samples'] or rss is None:failure='memory measurements unavailable'
 elif rss>budget*(1<<30) or memory['sampled_peak']['phys_footprint']>budget*(1<<30):failure='RSS or physical footprint exceeds budget'
 elif any(row.get('metal_peak_since_load',0)>budget*(1<<30) or row.get('reserved_bytes',0)>budget*(1<<30) for row in rows):failure='native allocation exceeds budget'
 return {'name':name,'command':[str(x) for x in cmd],'before':before,'after':environment(),'exit':code,'failure':failure,'seconds':time.monotonic()-start,'peak_rss_bytes':rss,'memory':memory,'samples':rows}

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','prefix']);p.add_argument('output',type=Path);p.add_argument('--budget-gib',type=int,required=True);p.add_argument('--draft');p.add_argument('--draft-vocab');p.add_argument('--draft-tokens',type=int,default=3);a=p.parse_args()
 r=Path(__file__).resolve().parents[1];e=r/'evidence/qwen-mtp-20261013';out=a.output;out.mkdir();q=str(Path.home()/'Models/Strata/Qwen3.8-IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf')
 env=os.environ.copy()
 for k in ['QWEN_DRAFT','DRAFT_TOKENS','DRAFT_VOCAB','PREFIX_CACHE']:env.pop(k,None)
 env['STRATA_SSD_EXPERTS']='1' if a.mode=='calibrate' else '0'
 if a.draft:env['QWEN_DRAFT']=a.draft;env['DRAFT_TOKENS']=str(a.draft_tokens)
 if a.draft_vocab:env['DRAFT_VOCAB']=a.draft_vocab
 files=[Path(__file__),r/'tools/bench_qwen_pair.py',r/'tools/apple_process_memory.py',e/'model-sha256.json'];jobs=[]
 if a.mode=='calibrate':
  oracle=e/'qwen-budget-oracle';files+=[oracle];configs=[{'slots':s,'batch':128 if a.budget_gib==8 else 1024} for s in ([32,24,16] if a.budget_gib==8 else [128,96,64])]
  files.extend(e/('workloads/chinese-%d.u32'%n) for n in [1024,4096]);linked=json.loads((e/'reference-linked-libraries.json').read_text())['files'];files.extend(Path(v) for v in linked)
  freeze={str(z):sha(z) for z in files}
  (out/'protocol.json').write_text(json.dumps({'mode':a.mode,'budget_gib':a.budget_gib,'configs':configs,'selection':'Test every predeclared slots/batch configuration on 1K and 4K, cap16, warmup+one hot. Exclude failed/budget-exceeding configurations; select lowest geometric complete-request time across both lengths. These are calibration, not formal speed results; validate selection with new six-pair runs.','sha256':freeze,'reference_precision':'F32 KV, flash off, same target; SSD expert file reads enabled'},indent=2)+'\n')
  candidates=[]
  for c in configs:
   selected=[]
   for n in [1024,4096]:
    cmd=[oracle,q,e/('workloads/chinese-%d.u32'%n),str(n+256),'16','1',str(c['slots']),str(c['batch'])]
    row=execute(out,'slots%d-batch%d-%d'%(c['slots'],c['batch'],n),cmd,env,a.budget_gib);jobs.append(row);(out/'runs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    if not row['failure'] and (len(row['samples'])!=2 or (row['samples'][0]['tokens'],row['samples'][0]['stop'])!=(row['samples'][1]['tokens'],row['samples'][1]['stop'])):row['failure']='missing/repeated result mismatch'
    selected.append(row);print(json.dumps({k:row[k] for k in ['name','exit','failure','seconds']}),flush=True)
   eligible=all(v['failure'] is None for v in selected);candidates.append({**c,'eligible':eligible,'geometric_total_s':math.exp(sum(math.log(v['samples'][1]['total_s']) for v in selected)/2) if eligible else None})
  valid=[v for v in candidates if v['eligible']];result={'candidates':candidates,'selected':min(valid,key=lambda v:v['geometric_total_s']) if valid else None,'qualified_comparison_available':bool(valid)}
 else:
  binary=Path.home()/'.cache/ingot-qwen-extended-measurement-20261013/release/examples/bench_sequence';m=json.loads((e/'conversation-v1/manifest.json').read_text());prompts=[Path(m['warmup'])]+[Path(v['file']) for v in m['rounds']];files += [binary,e/'extended-measurement-provenance.json',e/'conversation-v1/manifest.json',*prompts];freeze={str(z):sha(z) for z in files}
  (out/'protocol.json').write_text(json.dumps({'mode':a.mode,'budget_gib':a.budget_gib,'draft':a.draft,'draft_vocab':a.draft_vocab,'draft_tokens':a.draft_tokens,'pairs':6,'order':'AB BA AB BA AB BA; A prefix off, B prefix on','cap':256,'warmup':'One distinct input per process, excluded; first measured round must miss, later rounds share a fixed prefix.','primary':'sum of four complete request times excluding model load; honor EOS','ci':'paired block bootstrap 10000 resamples seed20261013; four correlated rounds stay in each process block','sha256':freeze},indent=2)+'\n')
  for pair in range(6):
   for arm in (['A','B'] if pair%2==0 else ['B','A']):
    cmd=[binary,q,str(a.budget_gib),'2560','256','off' if arm=='A' else 'on',*prompts];row=execute(out,'pair%d-%s'%(pair,arm),cmd,env,a.budget_gib);row.update(pair=pair,arm=arm)
    if row['failure'] is None:
     if len(row['samples'])!=5:row['failure']='missing round metrics'
     elif row['samples'][1]['prefix_reused_tokens']!=0:row['failure']='first measured request unexpectedly reused warmup'
     elif arm=='B' and any(s['prefix_reused_tokens']<=0 for s in row['samples'][2:]):row['failure']='later conversation did not hit prefix'
     elif arm=='A' and any(s['prefix_reused_tokens'] for s in row['samples']):row['failure']='cache-off unexpectedly reused prefix'
    if row['failure'] is None and jobs:
     if [(s['tokens'],s['stop']) for s in row['samples'][1:]]!=[(s['tokens'],s['stop']) for s in jobs[0]['samples'][1:]]:row['failure']='output or stop differs across cache arms/repetitions'
    jobs.append(row);(out/'runs.json').write_text(json.dumps(jobs,indent=2)+'\n');print(json.dumps({k:row[k] for k in ['name','exit','failure','seconds']}),flush=True)
    if row['failure']:raise RuntimeError(row['failure'])
  import numpy as np
  ratios=[];rounds=[]
  for pair in range(6):
   ar=next(v['samples'][1:] for v in jobs if v['pair']==pair and v['arm']=='A');br=next(v['samples'][1:] for v in jobs if v['pair']==pair and v['arm']=='B');ratios.append(sum(x['total_s'] for x in br)/sum(x['total_s'] for x in ar));rounds.append([b['total_s']/a['total_s'] for a,b in zip(ar,br)])
  indices=np.random.default_rng(20261013).integers(0,6,size=(10000,6));values=np.array(ratios);boot=np.exp(np.log(values)[indices].mean(axis=1));result={'paired_total_B_over_A':ratios,'paired_round_B_over_A':rounds,'geometric_total_ratio':float(np.exp(np.log(values).mean())),'ci95':np.quantile(boot,[.025,.975]).tolist(),'all_outputs_and_stops_exact':True,'every_process_within_memory_budget':True}
 assert all(sha(path)==v for path,v in freeze.items()),'measurement inputs changed'
 (out/'runs.json').write_text(json.dumps(jobs,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
