"""Frozen Qwen quality checks. Uses official answers and isolated executable tests."""
import argparse,collections,decimal,hashlib,json,math,os,re,resource,secrets,signal,subprocess,sys,tempfile,unicodedata
from pathlib import Path

def read_json(path):return json.loads(Path(path).read_text())
def read_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def normalize_zh(s):return ''.join(c.lower() for c in s if not c.isspace() and not unicodedata.category(c).startswith('P'))
def char_lcs_f1(answer, reference):
    a,b=normalize_zh(answer),normalize_zh(reference)
    if not a or not b:return float(a==b)
    previous=[0]*(len(b)+1)
    for x in a:
        current=[0]
        for j,y in enumerate(b):current.append(previous[j]+1 if x==y else max(previous[j+1],current[-1]))
        previous=current
    return 2.*previous[-1]/(len(a)+len(b))

def math_answer(s):
    marker='####' in s;tail=s.rsplit('####',1)[-1] if marker else s
    values=re.findall(r'(?<![\w.])[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)',tail)
    if not values:return None,marker
    try:return decimal.Decimal((values[0] if marker else values[-1]).replace(',','')),marker
    except decimal.InvalidOperation:return None,marker

def code_text(s):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',s,re.S|re.I)
    return '\n'.join(blocks) if blocks else s

def execute_python(source):
    with tempfile.TemporaryDirectory(prefix='qwen-eval-') as temp:
        temp=str(Path(temp).resolve());f=Path(temp)/'candidate.py';marker=('QWEN_TESTS_COMPLETED_'+secrets.token_hex(16)).encode();f.write_text(source+'\n\n__import__("os").write(1, '+repr(b'\n'+marker+b'\n')+')\n')
        policy='(version 1)(deny default)(allow process-exec (subpath "/Library/Developer/CommandLineTools"))(allow sysctl-read)(allow file-read-metadata)(allow file-read* (literal "/"))(allow file-read* (subpath "/dev") (subpath "/System") (subpath "/usr") (subpath "/Library/Developer/CommandLineTools") (subpath '+json.dumps(temp)+'))(allow file-read* file-write* (literal "/dev/null") (subpath '+json.dumps(temp)+'))'
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU,(3,3));resource.setrlimit(resource.RLIMIT_FSIZE,(1<<20,1<<20));resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
        # Output goes to capped files rather than unbounded PIPE buffering.
        with (Path(temp)/'stdout').open('w+b') as out,(Path(temp)/'stderr').open('w+b') as err:
            p=subprocess.Popen(['/usr/bin/sandbox-exec','-p',policy,sys.executable,'-I',str(f)],cwd=temp,env={'PATH':'/usr/bin:/bin','TMPDIR':temp,'HOME':temp},stdin=subprocess.DEVNULL,stdout=out,stderr=err,preexec_fn=limits,start_new_session=True)
            import time
            deadline=time.monotonic()+5;reason=None
            while reason is None:
                try:p.wait(timeout=.05);reason='ok' if p.returncode==0 else 'exit:'+str(p.returncode)
                except subprocess.TimeoutExpired:
                    rss=subprocess.run(['/bin/ps','-o','rss=','-p',str(p.pid)],capture_output=True,text=True).stdout.strip()
                    if rss and int(rss)>512*1024:reason='rss-limit'
                    elif time.monotonic()>deadline:reason='timeout'
                    if reason is not None:os.killpg(p.pid,signal.SIGKILL);p.wait()
            if reason=='ok':
                size=out.seek(0,2);out.seek(max(0,size-256))
                if marker not in out.read():reason='tests-not-completed'
            err.seek(0);detail=err.read(1000).decode(errors='replace')
            return {'passed':reason=='ok','reason':reason,'detail':detail}

def check_fixtures(manifest,out):
    with tempfile.TemporaryDirectory(prefix='qwen-denied-') as outside:
        denied = str(Path(outside).resolve()/'probe')
        probes={'ordinary':'assert sum([1,2,3]) == 6','home_read':'import os\nos.listdir('+repr(str(Path.home()))+')','network':'import socket\nsocket.socket(socket.AF_INET,socket.SOCK_DGRAM).sendto(b"probe",("127.0.0.1",9))','write_outside':'open('+repr(denied)+', "x").write("x")'}
        results={name:execute_python(code) for name,code in probes.items()}
        if not results['ordinary']['passed'] or any(results[n]['passed'] for n in probes if n!='ordinary'):raise RuntimeError(('sandbox failed',results))
    fixtures=[]
    for task in manifest['tasks']:
        if task['category']!='code':continue
        code=task['setup']+'\n'+task['reference_code']+'\n'+'\n'.join(task['tests'])
        result=execute_python(code);fixtures.append({'id':task['id'],**result})
    Path(out).write_text(json.dumps({'sandbox':results,'fixtures':fixtures},ensure_ascii=False,indent=2)+'\n')
    print('reference fixtures',sum(x['passed'] for x in fixtures),'/',len(fixtures),flush=True)

def compare_tf(manifest,reference,candidate,out):
    import numpy as np
    reference=Path(reference);candidate=Path(candidate);r={(x['category'],x['id']):x for x in read_rows(reference/'scores.jsonl')};c={(x['category'],x['id']):x for x in read_rows(candidate/'scores.jsonl')}
    if set(r)!=set(c) or len(r)!=len(manifest['teacher_forcing']):raise RuntimeError('incomplete/different TF documents')
    summary={};details=[];allkl=[];alltop=[];alloverlap=[];first=None
    for cat in ['zh','en','code','math','rare','total']:
        keys=[k for k in r if cat=='total' or k[0]==cat];nr=sum(r[k]['nll_sum'] for k in keys);nc=sum(c[k]['nll_sum'] for k in keys);tokens=sum(r[k]['tokens'] for k in keys)
        assert all(r[k]['tokens']==c[k]['tokens'] for k in keys)
        ratio=math.exp((nc-nr)/tokens);summary[cat]={'tokens':tokens,'reference_ppl':math.exp(nr/tokens),'candidate_ppl':math.exp(nc/tokens),'ppl_ratio':ratio,'passed':ratio<=(1.01 if cat=='total' else 1.02)}
    for doc in manifest['teacher_forcing']:
        name=doc['category']+'-'+doc['id']+'.f32';a=np.fromfile(reference/name,dtype='<f4').reshape(len(doc['full_logits_positions']),-1).astype(np.float64);b=np.fromfile(candidate/name,dtype='<f4').reshape(a.shape).astype(np.float64)
        assert a.shape[1]==248320 and np.isfinite(a).all() and np.isfinite(b).all()
        am=a.max(axis=1,keepdims=True);bm=b.max(axis=1,keepdims=True);la=a-am-np.log(np.exp(a-am).sum(axis=1,keepdims=True));lb=b-bm-np.log(np.exp(b-bm).sum(axis=1,keepdims=True));kl=(np.exp(la)*(la-lb)).sum(axis=1);top=a.argmax(axis=1)==b.argmax(axis=1)
        for i,pos in enumerate(doc['full_logits_positions']):
            ia=np.argpartition(a[i],-32)[-32:];ib=np.argpartition(b[i],-32)[-32:];overlap=len(set(ia)&set(ib))/32
            margin=np.sort(a[i,ia])[-1]-np.sort(a[i,ia])[-2]
            row={'category':doc['category'],'id':doc['id'],'position':pos,'kl':float(kl[i]),'top1_equal':bool(top[i]),'top32_overlap':overlap,'reference_margin':float(margin),'max_abs':float(np.max(np.abs(a[i]-b[i]))),'relative_l2':float(np.linalg.norm(a[i]-b[i])/max(np.linalg.norm(a[i]),1e-30))}
            details.append(row);allkl.append(float(kl[i]));alltop.append(bool(top[i]));alloverlap.append(overlap)
            if first is None and not top[i]:first=row
    result={'nll':summary,'samples':len(details),'kl_mean':float(np.mean(allkl)),'kl_p95':float(np.percentile(allkl,95)),'kl_max':max(allkl),'top1_fraction':float(np.mean(alltop)),'top32_overlap_mean':float(np.mean(alloverlap)),'first_top1_divergence':first,'details':details,'ppl_gate_passed':all(x['passed'] for x in summary.values())}
    Path(out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False),flush=True)

def paired_bound(differences):
    from scipy.stats import beta
    n=len(differences);plus=sum(x==1 for x in differences);minus=sum(x==-1 for x in differences);alpha=.0125
    low=0. if plus==0 else float(beta.ppf(alpha,plus,n-plus+1));high=1. if minus==n else float(beta.ppf(1-alpha,minus+1,n-minus))
    return {'n':n,'candidate_only_correct':plus,'reference_only_correct':minus,'difference':sum(differences)/n,'one_sided_lower':low-high,'passed':low-high>=-.02}

def score_tasks(manifest,reference,candidate,fixtures,out):
    r={(x['category'],x['id']):x for x in read_rows(reference)};c={(x['category'],x['id']):x for x in read_rows(candidate)}
    if set(r)!=set(c):raise RuntimeError('task ID mismatch')
    valid={x['id'] for x in read_json(fixtures)['fixtures'] if x['passed']};rows=[]
    for task in manifest['tasks']:
        key=(task['category'],task['id'])
        if key not in r:continue
        if key[0]=='code' and key[1] not in valid:continue
        pair=[]
        for generation in [r[key],c[key]]:
            text=generation['text'];detail={'stop':generation['stop'],'truncated':generation['stop']=='Length'}
            if key[0]=='zh':
                detail['correct']=normalize_zh(text) in [normalize_zh(a) for a in task['answers']]
                detail['char_lcs_f1']=max(char_lcs_f1(text,a) for a in task['answers'])
            elif key[0]=='math':value,marker=math_answer(text);detail.update(correct=value in [decimal.Decimal(a.replace(',','')) for a in task['answers']],format_ok=marker)
            else:detail.update(execute_python(task['setup']+'\n'+code_text(text)+'\n'+'\n'.join(task['tests'])));detail['correct']=detail['passed']
            pair.append(detail)
        rows.append({'category':key[0],'id':key[1],'reference':pair[0],'candidate':pair[1]})
    summary={}
    for category in ['zh','math','code','total']:
        subset=[x for x in rows if category=='total' or x['category']==category];d=[int(x['candidate']['correct'])-int(x['reference']['correct']) for x in subset]
        if not subset:
            summary[category]={'n':0,'passed':False,'reason':'category not yet evaluated'};continue
        summary[category]={**paired_bound(d),'reference_accuracy':sum(x['reference']['correct'] for x in subset)/len(subset),'candidate_accuracy':sum(x['candidate']['correct'] for x in subset)/len(subset),'reference_truncated':sum(x['reference']['truncated'] for x in subset),'candidate_truncated':sum(x['candidate']['truncated'] for x in subset)}
        if category=='zh':
            for side in ['reference','candidate']:summary[category][side+'_char_lcs_f1']=sum(x[side]['char_lcs_f1'] for x in subset)/len(subset)
    Path(out).write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['fixtures','tf','tasks']);parser.add_argument('manifest');parser.add_argument('paths',nargs='+');a=parser.parse_args();m=read_json(a.manifest)
    {'fixtures':check_fixtures,'tf':compare_tf,'tasks':score_tasks}[a.command](m,*a.paths)
