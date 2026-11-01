"""Compare a single trained GLM NextN layer using identical target hidden inputs.
These are numerical diagnostics, not target task-quality acceptance or head accuracy.
"""
from pathlib import Path
import argparse,array,hashlib,json,math

def values(path):
    raw=path.read_bytes()
    if len(raw)%4:raise ValueError('Invalid float32 dump: '+str(path))
    v=array.array('f');v.frombytes(raw)
    if not v or not all(map(math.isfinite,v)):raise ValueError('Empty/nonfinite tensor: '+str(path))
    return v

def stats(a,b):
    if len(a)!=len(b):raise ValueError('Different tensor lengths')
    delta=[x-y for x,y in zip(a,b)];norm=sum(x*x for x in a)
    return {'elements':len(a),'max_abs':max(map(abs,delta)),'rmse':math.sqrt(sum(x*x for x in delta)/len(a)),'relative_l2':math.sqrt(sum(x*x for x in delta)/max(norm,1e-30)),'reference_max_abs':max(map(abs,a))}

def main():
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    maps={name:(name+'-45.bin',name+'-45.bin') for name in ['mtp_tok_embd','mtp_enorm','mtp_hnorm','mtp_eh_proj','mtp_attn_norm','mtp_attn_out','mtp_ffn_inp','mtp_ffn_norm','mtp_ffn_out','mtp_post_ffn']}
    maps['mtp_shared_head_norm']=('mtp_shared_head_norm-45.bin','mtp_shared_head_norm.bin');maps['logits']=('mtp_logits-45.bin','result_output.bin')
    positions=sorted((d for d in a.native.iterdir() if d.is_dir() and d.name.isdigit()),key=lambda x:int(x.name))
    if not positions:raise ValueError('No positions')
    rows=[]
    for directory in positions:
        metrics={}
        for name,(native,reference) in maps.items():
            source=values(a.reference/directory.name/reference);candidate=values(directory/native);m=stats(source,candidate)
            if name=='logits':
                best=max(range(len(source)),key=source.__getitem__);actual=max(range(len(candidate)),key=candidate.__getitem__);rank=sorted(range(len(source)),key=source.__getitem__,reverse=True)
                m.update(reference_argmax=best,native_argmax=actual,top1_equal=best==actual,reference_margin=source[rank[0]]-source[rank[1]],top32_overlap=len(set(rank[:32])&set(sorted(range(len(candidate)),key=candidate.__getitem__,reverse=True)[:32]))/32)
            metrics[name]=m
        rows.append({'position':int(directory.name),'tensors':metrics})
    report={'scope':'Independent single NextN layer given identical native target hidden. Not an independent check of target hidden or end-task quality.','native':str(a.native),'reference':str(a.reference),'positions':rows,'all_head_argmax_equal':all(x['tensors']['logits']['top1_equal'] for x in rows),'worst_relative_l2':{name:max(x['tensors'][name]['relative_l2'] for x in rows) for name in maps},'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['all_head_argmax_equal','worst_relative_l2']}))
if __name__=='__main__':main()
