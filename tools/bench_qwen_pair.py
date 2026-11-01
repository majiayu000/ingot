"""Six frozen AB/BA blocks; raw failures and warmups retained. Single GPU only."""
from pathlib import Path
import argparse,hashlib,json,math,os,re,signal,subprocess,time
from apple_process_memory import MemoryMonitor

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def output_alignment(a,b):
 first=next((i for i,(x,y) in enumerate(zip(a['tokens'],b['tokens'])) if x!=y),None)
 if first is None and len(a['tokens'])!=len(b['tokens']):first=min(len(a['tokens']),len(b['tokens']))
 return {'equal':first is None and a['stop']==b['stop'],'first_different_token_index':first,'A_output_tokens':len(a['tokens']),'B_output_tokens':len(b['tokens']),'A_stop':a['stop'],'B_stop':b['stop']}
def environment():
 result={}
 for label,args in [('memory',['/usr/bin/vm_stat']),('processes',['/bin/ps','-axo','pid,comm,%cpu,rss']),('swap',['/usr/sbin/sysctl','vm.swapusage']),('thermal',['/usr/bin/pmset','-g','therm'])]:
  x=subprocess.run(args,capture_output=True,text=True);result[label]={'exit':x.returncode,'text':x.stdout+x.stderr}
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('evidence');parser.add_argument('candidate');parser.add_argument('output');parser.add_argument('--budget-gib',type=int,default=48);parser.add_argument('--baseline-native');parser.add_argument('--draft');parser.add_argument('--draft-vocab');parser.add_argument('--draft-tokens',type=int,default=3);parser.add_argument('--long',action='store_true');parser.add_argument('--reference-binary',type=Path);parser.add_argument('--reference-slots',type=int,default=0);parser.add_argument('--reference-batch',type=int,default=1024);a=parser.parse_args();e=Path(a.evidence);out=Path(a.output);out.mkdir(exist_ok=False)
 model=str(Path.home()/'Models/Strata/Qwen3.8-IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf');oracle=Path(a.baseline_native) if a.baseline_native else (a.reference_binary or e/'qwen-bench-oracle');cases=[(cat,n,e/f'workloads/{cat}-{n}.u32') for n in ([8192] if a.long else [1024,4096]) for cat in (['chinese','code'] if a.long else ['chinese','code','math'])]
 protocol={'pairs':6,'max_pairs':6,'order':'AB BA AB BA AB BA; A '+('frozen native AR baseline' if a.baseline_native else 'independent F32-KV AR reference')+'; B native','case_weights':str(len(cases))+' equal weights','primary':'complete request seconds excluding load; honor EOS; cap256','decode':'(output_count-1)/(total_s-first_token_s)','ci':'paired block bootstrap,10000 resamples,seed20261013; no production percentile claims','memory':f'native {a.budget_gib}GiB total reservation and RSS gate; legacy native Metal peak is unavailable and explicitly null; non-native reference is full resident with observed RSS, without an OS-wide reference RSS cap','reference_slots':a.reference_slots,'reference_batch':a.reference_batch,'reference_memory_cap_gib':a.budget_gib if a.reference_slots else None,'baseline_native':a.baseline_native,'candidate':str(Path(a.candidate).resolve()),'candidate_sha256':digest(a.candidate),'oracle_sha256':digest(oracle),'draft':a.draft,'draft_vocabulary':a.draft_vocab,'draft_tokens':a.draft_tokens,'model_hash_manifest_sha256':digest(e/'model-sha256.json'),'workloads':[{'category':c,'length':n,'path':str(p),'sha256':digest(p),'actual_tokens':p.stat().st_size//4} for c,n,p in cases],'invalid_conditions':['process failure','timeout1800s','missing metrics','repeated token/stop mismatch','budgeted process GPU or RSS peak exceeds requested budget'],'notes':['All timing samples and stderr retained; no selective replacement.','Draft-enabled B vs AR A is a combined-system comparison, not an AR kernel-only comparison.','Shared OS page cache and unrelated system work are not controlled; snapshots accompany every process.']}
 same_target=a.baseline_native is not None and Path(a.baseline_native).resolve()==Path(a.candidate).resolve()
 protocol['paired_output_contract']='Same native binary AR/MTP must produce identical token IDs and stop. Other arithmetic/backend comparisons record the first token divergence and output lengths without claiming exact equivalence.'
 protocol['paired_output_exact_required']=same_target
 protocol['harness_sha256']={str(z):digest(z) for z in [Path(__file__),Path(__file__).with_name('apple_process_memory.py')]}
 linked=json.loads((e/'reference-linked-libraries.json').read_text())['files'] if not a.baseline_native else {}
 for path,row in linked.items():
  if digest(path)!=row['sha256']:raise RuntimeError('Reference library changed: '+path)
 protocol['reference_libraries']=linked
 if a.reference_slots:protocol['memory']=f'Both processes limited to observed RSS/physical-footprint <= {a.budget_gib}GiB; native reservation/Metal peak also checked. Reference slots and batch calibrated before formal runs; this is a working budget configuration, not proof of optimal cache tuning.'
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records=[]
 for pair in range(6):
  for arm in (['A','B'] if pair%2==0 else ['B','A']):
   for cat,n,prompt in cases:
    tag=f'pair{pair}-{arm}-{cat}-{n}';env=os.environ.copy()
    for key in ['QWEN_DRAFT','DRAFT_VOCAB','DRAFT_TOKENS','PREFIX_CACHE']:env.pop(key,None)
    env['STRATA_SSD_EXPERTS']='1' if arm=='A' and a.reference_slots else '0'
    if arm=='A':
     cmd=([str(oracle),model,str(a.budget_gib),str(n+256),'256','1',str(prompt)] if a.baseline_native else [str(oracle),model,str(prompt),str(n+256),'256','1',str(a.reference_slots)]+([str(a.reference_batch)] if a.reference_binary else []))
    else:
     cmd=[a.candidate,model,str(a.budget_gib),str(n+256),'256','1',str(prompt)]
     if a.draft:env['QWEN_DRAFT']=a.draft;env['DRAFT_TOKENS']=str(a.draft_tokens)
     if a.draft_vocab:env['DRAFT_VOCAB']=a.draft_vocab
    before=environment();start=time.monotonic();code=None;failure=None
    with (out/(tag+'.stdout')).open('w') as stdout,(out/(tag+'.stderr')).open('w') as stderr:
     process=subprocess.Popen(['/usr/bin/time','-l',*cmd],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
     monitor=MemoryMonitor(process.pid,True)
     try:code=process.wait(timeout=1800)
     except subprocess.TimeoutExpired:
      failure='timeout1800s';os.killpg(process.pid,signal.SIGTERM)
      try:code=process.wait(timeout=10)
      except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
     finally:memory=monitor.close()
    record={'pair':pair,'arm':arm,'category':cat,'nominal_input':n,'command':cmd,'before':before,'after':environment(),'process_seconds':time.monotonic()-start,'exit':code,'failure':failure,'process_memory':memory}
    samples=[json.loads(line) for line in (out/(tag+'.stdout')).read_text().splitlines() if line.startswith('{')]
    record['warmup']=samples[0] if samples else None;record['sample']=samples[1] if len(samples)>1 else None
    match=re.search(r'(\d+)\s+maximum resident set size',(out/(tag+'.stderr')).read_text());record['peak_rss_bytes']=int(match[1]) if match else None
    if code!=0:record['failure']=record['failure'] or 'process failure'
    elif len(samples)!=2:record['failure']='missing metrics'
    elif memory['errors'] or not memory['samples']:record['failure']='process memory accounting unavailable'
    elif (samples[0]['tokens'],samples[0]['stop'])!=(samples[1]['tokens'],samples[1]['stop']):record['failure']='repeated token/stop mismatch'
    elif (arm=='B' or a.baseline_native or a.reference_slots) and ((samples[1].get('metal_peak_since_load') is not None and samples[1]['metal_peak_since_load']>a.budget_gib*(1<<30)) or record['peak_rss_bytes'] is None or record['peak_rss_bytes']>a.budget_gib*(1<<30)):record['failure']='budgeted process GPU or RSS peak exceeds requested budget'
    elif (arm=='B' or a.baseline_native or a.reference_slots) and memory['sampled_peak']['phys_footprint']>a.budget_gib*(1<<30):record['failure']='budgeted process sampled physical footprint exceeds requested budget'
    if not record['failure']:
     prior=next((v['sample'] for v in records if v['arm']==arm and v['category']==cat and v['nominal_input']==n),None)
     if prior is not None and (prior['tokens'],prior['stop'])!=(samples[1]['tokens'],samples[1]['stop']):record['failure']='greedy output/stop differs across process repetitions'
    if not record['failure']:
     peer=next((v for v in records if v['pair']==pair and v['arm']!=arm and v['category']==cat and v['nominal_input']==n),None)
     if peer is not None:
      aligned=output_alignment(peer['sample'],samples[1]) if arm=='B' else output_alignment(samples[1],peer['sample'])
      record['paired_output_alignment']=aligned
      if same_target and not aligned['equal']:record['failure']='AR/MTP token or stop mismatch for the same target binary'
    records.append(record);(out/'runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['pair','arm','category','nominal_input','exit','failure','process_seconds']}),flush=True)
    if record['failure']:raise RuntimeError(record['failure'])
 for path,row in linked.items():
  if digest(path)!=row['sha256']:raise RuntimeError('Reference library changed during run: '+path)
 if digest(a.candidate)!=protocol['candidate_sha256'] or digest(oracle)!=protocol['oracle_sha256']:raise RuntimeError('Binary changed during run')
 for row in protocol['workloads']:
  if digest(row['path'])!=row['sha256']:raise RuntimeError('Workload changed during run')
 import numpy as np
 rng=np.random.default_rng(20261013);samples=rng.integers(0,6,size=(10000,6));report={};ratios=[]
 for cat,n,_ in cases:
  arows=[next(r['sample'] for r in records if r['pair']==i and r['arm']=='A' and r['category']==cat and r['nominal_input']==n) for i in range(6)]
  brows=[next(r['sample'] for r in records if r['pair']==i and r['arm']=='B' and r['category']==cat and r['nominal_input']==n) for i in range(6)]
  ratio=np.array([b['total_s']/a['total_s'] for a,b in zip(arows,brows)]);ratios.append(ratio);boot=np.exp(np.log(ratio)[samples].mean(axis=1));row={'paired_B_over_A':ratio.tolist(),'geometric_ratio':float(np.exp(np.log(ratio).mean())),'ci95':np.quantile(boot,[.025,.975]).tolist()}
  for arm,rows in [('A',arows),('B',brows)]:
   total=np.array([r['total_s'] for r in rows]);decode_seconds=sum(r['total_s']-r['first_token_s'] for r in rows);row[arm]={'median_total_s':float(np.median(total)),'range_total_s':[float(total.min()),float(total.max())],'sample_variance_s2':float(np.var(total,ddof=1)),'weighted_decode_tokens_s':sum(max(0,len(r['tokens'])-1) for r in rows)/decode_seconds,'output_lengths':[len(r['tokens']) for r in rows],'ttft_s':[r['first_token_s'] for r in rows],'stops':[r['stop'] for r in rows]}
  row['paired_output_alignment']=[output_alignment(x,y) for x,y in zip(arows,brows)]
  report[f'{cat}-{n}']=row
 logs=np.log(np.array(ratios)).mean(axis=0);boot=np.exp(logs[samples].mean(axis=1));report['equal_weight_total']={'geometric_ratio':float(np.exp(logs.mean())),'ci95':np.quantile(boot,[.025,.975]).tolist(),'block_count':6}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['equal_weight_total']),flush=True)
if __name__=='__main__':main()
