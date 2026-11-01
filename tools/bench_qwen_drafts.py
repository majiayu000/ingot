"""Development ablation: two balanced blocks, exact greedy output, fixed cap.
Select a configuration for a separate six-pair confirmation; no significance claim.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,re,signal,statistics,subprocess,time
from apple_process_memory import MemoryMonitor
from bench_qwen_pair import environment

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('evidence');p.add_argument('candidate');p.add_argument('output');a=p.parse_args()
    e=Path(a.evidence);out=Path(a.output);out.mkdir(exist_ok=False)
    q=str(Path.home()/'Models/Strata/Qwen3.8-IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf')
    d=Path.home()/'Models/Strata/MTP';q8=d/'mtp-Qwen3.8-Flash-Next-Q8_0.gguf';q4=d/'mtp-Qwen3.8-Flash-Next-local-Q4_0.gguf';h4=d/'mtp-Qwen3.8-Flash-Next-local-head-Q4_0.gguf';compact=d/'mtp-Qwen3.8-Flash-Next-Q8_0-shortlist-75264.gguf';vocab=d/'mtp-Qwen3.8-Flash-Next-Q8_0-shortlist-75264.i32'
    configs=[('ar',None,0,None),('q8-k1',q8,1,None),('q8-k2',q8,2,None),('q8-k3',q8,3,None),('q4-k3',q4,3,None),('head-q4-k3',h4,3,None),('q4-shortlist-k3',q4,3,vocab),('head-q4-shortlist-k3',h4,3,vocab),('q8-shortlist-k3',q8,3,vocab),('compact-shortlist-k3',compact,3,vocab)]
    cases=[(cat,n,e/f'workloads/{cat}-{n}.u32') for n in [1024,4096] for cat in ['chinese','code','math']]+[('rare',1024,e/'workloads/rare-1024.u32')]
    files=[Path(a.candidate),Path(__file__),Path(__file__).with_name('bench_qwen_pair.py'),Path(__file__).with_name('apple_process_memory.py'),e/'model-sha256.json',e/'workloads/rare-1024-manifest.json']+[path for _,_,path in cases]
    freeze={str(path):sha(path) for path in files}
    protocol={'configs':[{'name':name,'draft':str(head) if head else None,'depth':depth,'vocabulary':str(v) if v else None} for name,head,depth,v in configs],'blocks':2,'order':'declared config order, then reverse; fixed case order within configuration','budget_gib':48,'prefix_cache':False,'cap':128,'input_weights':'six equal primary weights: Chinese/code/math at1K/4K; rare-token case reported separately','warmup':'round0 per process, retained and excluded from ranking','primary':'geometric mean of paired complete-request B/AR ratios across six primary cases and two blocks','selection':'smallest geometric mean, including AR. Separate six-pair confirmation is required; two blocks provide no acceptance CI.','invariants':'Every warm/hot token and stop must equal AR for the case; outputs may differ by case, never by draft configuration.','sha256':freeze,'cache_state':'warm or unknown OS file cache, not cold SSD'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records=[];expected={}
    for block,ordered in enumerate([configs,list(reversed(configs))]):
        for label,head,depth,v in ordered:
            for cat,n,path in cases:
                tag=f'block{block}-{label}-{cat}-{n}';env=os.environ.copy()
                for key in ['QWEN_DRAFT','DRAFT_TOKENS','DRAFT_VOCAB','PREFIX_CACHE']:env.pop(key,None)
                if head:env.update(QWEN_DRAFT=str(head),DRAFT_TOKENS=str(depth))
                if v:env['DRAFT_VOCAB']=str(v)
                cmd=[a.candidate,q,'48',str(n+256),'128','1',str(path)]
                before=environment();start=time.monotonic();failure=None
                with (out/(tag+'.stdout')).open('x') as stdout,(out/(tag+'.stderr')).open('x') as stderr:
                    process=subprocess.Popen(['/usr/bin/time','-l',*cmd],env=env,stdout=stdout,stderr=stderr,start_new_session=True);monitor=MemoryMonitor(process.pid,True)
                    try:code=process.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        failure='timeout1800s';os.killpg(process.pid,signal.SIGTERM)
                        try:code=process.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
                    finally:memory=monitor.close()
                samples=[json.loads(line) for line in (out/(tag+'.stdout')).read_text().splitlines() if line.startswith('{')]
                match=re.search(r'(\d+)\s+maximum resident set size',(out/(tag+'.stderr')).read_text());rss=int(match[1]) if match else None
                record={'block':block,'config':label,'category':cat,'nominal_input':n,'command':cmd,'exit':code,'failure':failure,'seconds':time.monotonic()-start,'before':before,'after':environment(),'process_memory':memory,'peak_rss_bytes':rss,'samples':samples}
                if code:record['failure']=failure or 'process failure'
                elif len(samples)!=2:record['failure']='missing warm/hot metrics'
                elif memory['errors'] or not memory['samples']:record['failure']='process memory accounting unavailable'
                elif rss is None or rss>48*(1<<30) or memory['sampled_peak']['phys_footprint']>48*(1<<30) or any(x['metal_peak_since_load']>48*(1<<30) for x in samples):record['failure']='memory budget exceeded'
                else:
                    key=(cat,n)
                    if key not in expected:
                        assert label=='ar';expected[key]=(samples[0]['tokens'],samples[0]['stop'])
                    for sample in samples:
                        if (sample['tokens'],sample['stop'])!=expected[key]:record['failure']='output or stop differs from same-target AR'
                        if sample['prefix_reused_tokens']!=0:record['failure']='unexpected prefix reuse'
                records.append(record);(out/'runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['block','config','category','nominal_input','seconds','failure']}),flush=True)
                if record['failure']:raise RuntimeError(record['failure'])
    assert all(sha(p)==h for p,h in freeze.items()),'benchmark inputs or binaries changed'
    report={}
    for label,_,_,_ in configs:
        ratios=[];case_report={}
        for cat,n,_ in cases:
            rows=[next(x['samples'][1] for x in records if x['block']==i and x['config']==label and x['category']==cat and x['nominal_input']==n) for i in range(2)]
            refs=[next(x['samples'][1] for x in records if x['block']==i and x['config']=='ar' and x['category']==cat and x['nominal_input']==n) for i in range(2)]
            ratio=[b['total_s']/r['total_s'] for b,r in zip(rows,refs)]
            if cat!='rare':ratios.extend(ratio)
            rounds=[v for row in rows for v in row['mtp_rounds']];proposed=sum(v['proposed'] for v in rounds);accepted=sum(v['accepted'] for v in rounds)
            case_report[f'{cat}-{n}']={'paired_over_ar':ratio,'hot_total_s':[x['total_s'] for x in rows],'median_total_s':statistics.median(x['total_s'] for x in rows),'hot_prefill_s':[x['prefill_s'] for x in rows],'output_lengths':[len(x['tokens']) for x in rows],'stops':[x['stop'] for x in rows],'proposed':proposed,'accepted':accepted,'acceptance':accepted/proposed if proposed else None,'rounds':len(rounds),'committed':sum(v['committed'] for v in rounds),'verification_tokens':sum(v['verification_tokens'] for v in rounds),'replayed_tokens':sum(v['replayed_tokens'] for v in rounds),'round_cost_s':{k:sum(v[k] for v in rounds) for k in ['head_s','verify_s','snapshot_s','snapshot_encode_s','replay_s','catch_up_s','total_s']}}
        report[label]={'geometric_over_ar':math.exp(sum(math.log(v) for v in ratios)/len(ratios)),'cases':case_report}
    selected=min(report,key=lambda k:report[k]['geometric_over_ar'])
    (out/'summary.json').write_text(json.dumps({'development_screening_only':True,'all_outputs_equal_ar':True,'selected_for_confirmation':selected,'configurations':report},indent=2)+'\n');print(json.dumps({'selected_for_confirmation':selected,'ratio':report[selected]['geometric_over_ar']}),flush=True)
if __name__=='__main__':main()
