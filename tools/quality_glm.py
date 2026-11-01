"""GLM quality: same frozen Qwen scoring functions; model-specific vocabulary and thinking boundary."""
import argparse,json,math
from pathlib import Path
from quality_qwen import read_json,read_rows,score_tasks

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
        assert a.shape[1]==154880 and np.isfinite(a).all() and np.isfinite(b).all()
        am=a.max(axis=1,keepdims=True);bm=b.max(axis=1,keepdims=True);la=a-am-np.log(np.exp(a-am).sum(axis=1,keepdims=True));lb=b-bm-np.log(np.exp(b-bm).sum(axis=1,keepdims=True));kl=(np.exp(la)*(la-lb)).sum(axis=1);top=a.argmax(axis=1)==b.argmax(axis=1)
        for i,pos in enumerate(doc['full_logits_positions']):
            ia=np.argpartition(a[i],-32)[-32:];ib=np.argpartition(b[i],-32)[-32:];overlap=len(set(ia)&set(ib))/32
            margin=np.sort(a[i,ia])[-1]-np.sort(a[i,ia])[-2]
            row={'category':doc['category'],'id':doc['id'],'position':pos,'kl':float(kl[i]),'top1_equal':bool(top[i]),'top32_overlap':overlap,'reference_margin':float(margin),'max_abs':float(np.max(np.abs(a[i]-b[i]))),'relative_l2':float(np.linalg.norm(a[i]-b[i])/max(np.linalg.norm(a[i]),1e-30))}
            details.append(row);allkl.append(float(kl[i]));alltop.append(bool(top[i]));alloverlap.append(overlap)
            if first is None and not top[i]:first=row
    result={'nll':summary,'samples':len(details),'kl_mean':float(np.mean(allkl)),'kl_p95':float(np.percentile(allkl,95)),'kl_max':max(allkl),'top1_fraction':float(np.mean(alltop)),'top32_overlap_mean':float(np.mean(alloverlap)),'first_top1_divergence':first,'details':details,'ppl_gate_passed':all(x['passed'] for x in summary.values())}
    Path(out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False),flush=True)


def final_answers(source):
    source=Path(source);out=source.with_name(source.stem+'-final.jsonl');rows=[]
    for original in read_rows(source):
        row=dict(original);text=row['text'];closing='</think>' in text
        row['reasoning_closed']=closing
        row['text']=text.rsplit('</think>',1)[1].strip() if closing else ''
        rows.append(row)
    with out.open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['tf','tasks']);p.add_argument('manifest');p.add_argument('paths',nargs='+');a=p.parse_args();m=read_json(a.manifest)
    if a.command=='tf':compare_tf(m,*a.paths)
    else:
        reference,candidate,fixtures,out=a.paths
        score_tasks(m,final_answers(reference),final_answers(candidate),fixtures,out)
