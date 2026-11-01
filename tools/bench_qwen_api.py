"""Frozen natural-input server comparison. One owned server; no concurrent model job.
API E2E includes local HTTP. Engine-reported timing is retained separately. Never
infer per-token latency from MTP text chunks. Cross-quantization is explicit.
"""
from pathlib import Path
import argparse,hashlib,json,os,resource,signal,struct,subprocess,time,urllib.request
from apple_process_memory import MemoryMonitor

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def post(base,path,data):
 req=urllib.request.Request(base+path,json.dumps(data).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=1800) as response:return json.load(response)
def stream(base,path,data):
 req=urllib.request.Request(base+path,json.dumps(data).encode(),{'Content-Type':'application/json'});start=time.monotonic();first=None;events=[];done=False
 with urllib.request.urlopen(req,timeout=1800) as response:
  if 'text/event-stream' not in response.headers.get('Content-Type',''):raise RuntimeError('Expected SSE')
  for line in response:
   if not line.startswith(b'data:'):continue
   payload=line[5:].strip()
   if payload==b'[DONE]':done=True;break
   event=json.loads(payload);elapsed=time.monotonic()-start;events.append({'seconds':elapsed,'data':event})
   if event.get('error'):raise RuntimeError(json.dumps(event['error']))
   text=event.get('content') or ''.join(c.get('text') or '' for c in event.get('choices',[]))
   if text and first is None:first=elapsed
   if event.get('stop') is True:done=True;break
 if not done:raise RuntimeError('SSE ended without completion')
 return {'total_s':time.monotonic()-start,'first_visible_s':first,'events':events}
def snapshot():
 out={}
 for k,cmd in [('swap',['/usr/sbin/sysctl','vm.swapusage']),('memory',['/usr/bin/vm_stat']),('thermal',['/usr/bin/pmset','-g','therm'])]:
  p=subprocess.run(cmd,capture_output=True,text=True);out[k]={'exit':p.returncode,'text':p.stdout+p.stderr}
 return out

def quality_tasks(base,framework,model,e,out,reference):
 from strata_tokenizer import Tokenizer
 target=str(Path.home()/'Models/Strata/Qwen3.8-IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf')
 tk=Tokenizer.from_gguf(target);manifest=e/'quality-v1/manifest.json';m=json.loads(manifest.read_text());tasks={(v['category'],v['id']):v for v in m['tasks']}
 refs=[json.loads(line) for line in reference.read_text().splitlines()];ids=[(v['category'],v['id']) for v in refs]
 if len(set(ids))!=len(ids):raise RuntimeError('Duplicate reference task IDs')
 freeze={str(manifest):sha(manifest),str(reference):sha(reference)}
 for key in ids:
  t=tasks[key];path=e/'quality-v1'/t['file']
  if sha(path)!=t['sha256']:raise RuntimeError('Frozen task changed')
  freeze[str(path)]=t['sha256']
 (out/'quality-protocol.json').write_text(json.dumps({'framework':framework,'reference':str(reference),'n':len(ids),'sha256':freeze,'template':'exact frozen non-thinking raw completion; independently checked tokenizer IDs','generation':'temperature0, no repetition penalty, honor EOS, frozen per-task cap; one request per ID, no retry or score-driven expansion','scorer':'existing frozen quality_qwen.py; independent answers and sandboxed Python tests','comparison':'different oMLX quantization is a system quality comparison, not pure-engine parity'},indent=2)+'\n')
 with (out/'tasks-decoded.jsonl').open('x') as result,(out/'task-http-events.jsonl').open('x') as rawout:
  for i,key in enumerate(ids):
   t=tasks[key];raw=(e/'quality-v1'/t['file']).read_bytes();tokens=list(struct.unpack('<'+'I'*(len(raw)//4),raw));text=tk.decode(tokens)
   if framework=='strata':
    check=post(base,'/tokenize',{'content':text,'add_special':False})['tokens'];request={'prompt':tokens,'n_predict':t['cap'],'temperature':0,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,'seed':20261013,'cache_prompt':False,'ignore_eos':False,'stream':True};endpoint='/completion'
   else:
    check=post(base,'/tokenize',{'model':model,'prompt':text,'add_special_tokens':True})['tokens'];request={'model':model,'prompt':text,'max_tokens':t['cap'],'temperature':0,'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0,'seed':20261013,'stream':True,'stream_options':{'include_usage':True}};endpoint='/v1/completions'
   if check!=tokens:raise RuntimeError('Quality tokenizer mismatch: '+str(key))
   row=stream(base,endpoint,request);events=[v['data'] for v in row['events']]
   generated=''.join(v.get('content') or ''.join(c.get('text') or '' for c in v.get('choices',[])) for v in events)
   if framework=='strata':
    final=events[-1]
    if final.get('tokens_evaluated')!=len(tokens) or final.get('timings',{}).get('cache_n')!=0:raise RuntimeError('Quality prompt/cache count mismatch')
    reason=final.get('stop_type');count=final['tokens_predicted']-int(reason=='eos');stop='Eos' if reason=='eos' else 'Length' if reason=='limit' else None
   else:
    usage=next((v['usage'] for v in reversed(events) if v.get('usage')),None)
    if usage is None or usage['prompt_tokens']!=len(tokens) or (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0):raise RuntimeError('Quality usage/cache mismatch')
    reason=next((c['finish_reason'] for v in reversed(events) for c in v.get('choices',[]) if c.get('finish_reason')),None);count=usage['completion_tokens'];stop='Eos' if reason=='stop' else 'Length' if reason=='length' else None
   if stop is None:raise RuntimeError('Unexpected quality stop reason '+str(reason))
   rawout.write(json.dumps({'category':key[0],'id':key[1],**row},ensure_ascii=False)+'\n');rawout.flush()
   result.write(json.dumps({'category':key[0],'id':key[1],'text':generated,'stop':stop,'output_tokens':count,'total_s':row['total_s'],'first_visible_s':row['first_visible_s']},ensure_ascii=False)+'\n');result.flush()
   if (i+1)%10==0:print(json.dumps({'quality_completed':i+1,'total':len(ids),'category':key[0]}),flush=True)
 if any(sha(Path(path))!=value for path,value in freeze.items()):raise RuntimeError('Quality input drift')
 return {'completed':True,'n':len(ids),'decoded':str(out/'tasks-decoded.jsonl')}

def main():
 p=argparse.ArgumentParser();p.add_argument('framework',choices=['strata','omlx']);p.add_argument('output');p.add_argument('--depth',type=int,default=3);p.add_argument('--draft');p.add_argument('--strata-binary',type=Path);p.add_argument('--tasks-reference',type=Path);p.add_argument('--repeats',type=int,default=3);p.add_argument('--short',action='store_true',help='only the six 1K/4K cases for paired confirmation');a=p.parse_args()
 r=Path(__file__).resolve().parents[1];e=r/'evidence/qwen-mtp-20261013';x=r.parents[1]/'strata-comparison-20261010';out=Path(a.output);out.mkdir(exist_ok=False)
 port=next((n for n in range(5567,5587) if subprocess.run(['/usr/sbin/lsof','-nP',f'-iTCP:{n}','-sTCP:LISTEN'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==1),None)
 if port is None:raise RuntimeError('No free port in the assigned private test range')
 base=f'http://127.0.0.1:{port}';env=os.environ.copy();q=str(Path.home()/'Models/Strata/Qwen3.8-IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf')
 if a.framework=='strata':
  draft=a.draft or str(Path.home()/'Models/Strata/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf');binary=a.strata_binary or e/'strata-current-frozen/bin/llama-server'
  cmd=[str(binary),'--offline','-m',q,'-ngl','99','-c','8448','-b','1024','-ub','1024','-np','1','-lm','mmap','-lzm','on','-t','8','--host','127.0.0.1','--port',str(port),'--metrics']
  if a.depth:cmd+=['--spec-type','draft-mtp','--model-draft',draft,'--spec-draft-n-max',str(a.depth),'--spec-draft-p-min','0']
  env['STRATA_SSD_EXPERTS']='0';env['DYLD_LIBRARY_PATH']=str(binary.parent);model=q;health='/health';identity={str(p):sha(p) for p in binary.parent.iterdir() if p.name=='llama-server' or p.suffix=='.dylib'}
 else:
  models=out/'models';models.mkdir();(models/'omlx-native').symlink_to(x/'models/omlx-native',target_is_directory=True);settings=out/'settings';settings.mkdir();model='omlx-native'
  config={'mtp_enabled':bool(a.depth),'mtp_fixed_depth':a.depth or 1,'qwen4_ple_ssd_offload':True,'moe_expert_offload_enabled':False,'moe_expert_offload_resident_fraction':.25}
  (settings/'model_settings.json').write_text(json.dumps({'version':1,'models':{model:config}},indent=2)+'\n')
  identity_manifests=[e/'omlx-file-sha256.json',e/'omlx-runtime-sha256.json']
  binary=x/'.venv-omlx/bin/omlx';cmd=[str(binary),'serve','--model-dir',str(models),'--base-path',str(settings),'--host','127.0.0.1','--port',str(port),'--max-concurrent-requests','1','--memory-guard','aggressive','--no-cache','--headless'];env['OMLX_CACHE_ENABLED']='0';health='/v1/models';identity={str(z):sha(z) for z in [binary,x/'models/omlx-native/config.json',x/'models/omlx-native/tokenizer.json',x/'.venv-omlx/lib/python3.13/site-packages/omlx/server.py',*identity_manifests]}
 cases=[] if a.tasks_reference else [(c,n) for n in [1024,4096] for c in ['chinese','code','math']]+([] if a.short else [('chinese',8192),('code',8192)])
 protocol={'framework':a.framework,'command':cmd,'sha256':identity,'depth':a.depth,'round0':'per-case engine warmup excluded; all subsequent repeats retained','repeats':a.repeats,'cap':256,'eos':'honored','prefix_cache':False,'primary':'client E2E excluding model load','timing_note':'No per-token latency claims from streaming chunks; server timings and chunk times retained separately. This standalone comparator run is not a paired native/server speedup estimate.','weights':'Same GGUF target for Strata; oMLX mixed 4-bit affine checkpoint is different quantization, a usable-system comparison only.','samples':[],'before':snapshot(),'completed':False}
 protocol['harness_sha256']={str(z):sha(z) for z in [Path(__file__),Path(__file__).with_name('apple_process_memory.py')]}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');process=None;monitor=None
 try:
  with (out/'server.log').open('x') as log:
   process=subprocess.Popen(cmd,env=env,stdout=log,stderr=log,start_new_session=True)
   monitor=MemoryMonitor(process.pid)
   ready=time.monotonic()+600
   while True:
    if process.poll() is not None:raise RuntimeError('Server exited before ready')
    try:
     with urllib.request.urlopen(base+health,timeout=2) as response:
      if response.status==200:break
    except OSError:pass
    if time.monotonic()>ready:raise RuntimeError('Server startup timeout')
    time.sleep(2)
   for cat,length in cases:
    tokenfile=e/f'workloads/{cat}-{length}.u32';raw=tokenfile.read_bytes();tokens=list(struct.unpack('<'+'I'*(len(raw)//4),raw));text=(e/f'workloads/{cat}-{length}.rendered.txt').read_text();protocol['samples'].append({'category':cat,'nominal_tokens':length,'actual_tokens':len(tokens),'input_sha256':sha(tokenfile),'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
    if a.framework=='strata':
     check=post(base,'/tokenize',{'content':text,'add_special':False})['tokens'];request={'prompt':tokens,'n_predict':256,'temperature':0,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,'seed':20261013,'cache_prompt':False,'ignore_eos':False,'return_tokens':True,'stream':True};endpoint='/completion'
    else:
     check=post(base,'/tokenize',{'model':model,'prompt':text,'add_special_tokens':True})['tokens'];request={'model':model,'prompt':text,'max_tokens':256,'temperature':0,'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0,'seed':20261013,'stream':True,'stream_options':{'include_usage':True}};endpoint='/v1/completions'
    if check!=tokens:raise RuntimeError(f'Engine tokenizer mismatch for {cat}-{length}')
    previous=None
    for repeat in range(a.repeats+1):
     row=stream(base,endpoint,request);row.update(category=cat,nominal_tokens=length,input_tokens=len(tokens),repeat=repeat,request=request);events=[v['data'] for v in row['events']]
     row['text']=''.join(v.get('content') or ''.join(c.get('text') or '' for c in v.get('choices',[])) for v in events)
     if a.framework=='strata':
      row['stream_token_ids']=[t for v in events for t in v.get('tokens',[])];row['server']=events[-1];row['stop']=events[-1].get('stop_type')
      # This server counts EOG in n_gen and may suppress partial-UTF8 token chunks.
      # The final counter, not the number of streamed IDs, gives generated length.
      if row['server'].get('tokens_evaluated')!=len(tokens):raise RuntimeError('Strata prompt token count mismatch')
      if row['server'].get('timings',{}).get('cache_n')!=0:raise RuntimeError('Unexpected Strata prefix reuse')
      row['server_generated_including_eos']=row['server']['tokens_predicted'];row['output_tokens']=row['server_generated_including_eos']-int(row['stop']=='eos')
      row['output_count_definition']='final server n_gen minus one on EOS; streamed token IDs can omit partial-UTF8 pieces'
      current=(row['text'],row['stop'],row['output_tokens'])
     else:
      usage=next((v['usage'] for v in reversed(events) if v.get('usage')),None)
      if usage is None or usage['prompt_tokens']!=len(tokens):raise RuntimeError('Missing/mismatched usage')
      if (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0):raise RuntimeError('Unexpected prefix reuse')
      row['server']=usage;row['output_tokens']=usage['completion_tokens'];row['stop']=next((c['finish_reason'] for v in reversed(events) for c in v.get('choices',[]) if c.get('finish_reason')),None);current=(row['text'],row['stop'])
     (out/f'{cat}-{length}-{repeat}.json').write_text(json.dumps(row,ensure_ascii=False)+'\n')
     if previous is not None and previous!=current:raise RuntimeError('Repeated greedy output/stop differs')
     previous=current;print(json.dumps({k:row[k] for k in ['category','nominal_tokens','repeat','total_s','first_visible_s','output_tokens','stop']}),flush=True)
   if a.tasks_reference:protocol['quality_tasks']=quality_tasks(base,a.framework,model,e,out,a.tasks_reference)
   for path,expected in identity.items():
    if sha(path)!=expected:raise RuntimeError('Comparator runtime changed during measurement: '+path)
   for sample in protocol['samples']:
    tokenfile=e/f"workloads/{sample['category']}-{sample['nominal_tokens']}.u32"
    if sha(tokenfile)!=sample['input_sha256']:raise RuntimeError('Comparator input changed')
   protocol['completed']=True
 finally:
  if process is not None and process.poll() is None:
   os.killpg(process.pid,signal.SIGINT)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
  if monitor is not None:
   memory=monitor.close();(out/'process-memory.json').write_text(json.dumps(memory,indent=2)+'\n');protocol['sampled_process_peaks']=memory['sampled_peak'];protocol['memory_accounting_errors']=memory['errors']
  protocol['after']=snapshot();protocol['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
if __name__=='__main__':main()
