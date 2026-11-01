#!/usr/bin/env python3
"""Native formats/tokenizer oracle. Model inference oracles are separate C++ tools."""
import argparse,ast,ctypes,json,pathlib,subprocess,tempfile
import numpy as np
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--model',type=pathlib.Path,required=True);ap.add_argument('--ggml',type=pathlib.Path,required=True);ap.add_argument('--binary',type=pathlib.Path,default=pathlib.Path('target/release/ingot'));ap.add_argument('--report',type=pathlib.Path,required=True)
 ap.add_argument('--tokenizer-oracle',type=pathlib.Path,help='Pinned binary supporting the tokenize command; required for GLM')
 a=ap.parse_args();shards=sorted(a.model.parent.glob(a.model.name.replace('00001-of','*-of')));files=[GGUFFile(p) for p in shards];lib=ctypes.CDLL(str(a.ggml.resolve()));formats={}
 for g in files:
  for t in g.tensors:formats.setdefault(t.type_name,(g,t))
 if files[0].metadata.get('tokenizer.ggml.pre')=='glm4' and not a.tokenizer_oracle:ap.error('GLM requires --tokenizer-oracle')
 report={'formats':[],'tokenizer':[]}
 with tempfile.TemporaryDirectory() as tmp:
  for name,(g,t) in sorted(formats.items()):
   n=t.shape[0];stride=t.expected_bytes()//(t.elements//n)
   for row in sorted({0,min(7,t.elements//n-1)}):
    with g.path.open('rb') as f:f.seek(g.data_start+t.offset+row*stride);raw=f.read(stride)
    if name=='F32':expected=np.frombuffer(raw,dtype='<f4')
    elif name=='F16':expected=np.frombuffer(raw,dtype='<f2').astype(np.float32)
    elif name=='BF16':expected=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
    else:
     suffix=name.lower()[:-1]+'K' if name.endswith('_K') else name.lower()
     fn=getattr(lib,'dequantize_row_'+suffix);fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64];fn.restype=None
     expected=np.empty(n,np.float32);source=ctypes.create_string_buffer(raw);fn(source,expected.ctypes.data,n)
    dest=pathlib.Path(tmp)/f'{name}-{row}.f32';subprocess.run([str(a.binary),'dequant',str(a.model),t.name,str(row),str(dest)],check=True)
    actual=np.fromfile(dest,dtype='<f4');error=float(np.max(np.abs(actual-expected)))
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=1e-8);report['formats'].append({'format':name,'tensor':t.name,'row':row,'max_abs':error})
  tk=None if a.tokenizer_oracle else Tokenizer.from_gguf(a.model)
  corpus=['','Hello, world!','你好，世界！','def f(x):\n\treturn x + 123','e\u0301\u0301 _j','😀🚀','  \r\n\r\n\t trailing  ','1234567890','<|im_start|>assistant\n<think>\n</think>','[gMASK]<sop><|user|>你好<|assistant|><think>','<tool_response>','<|im_star','\x00\x01\x7f','مرحبا','\u2028\u2029','x'*4096]
  for special in [False,True]:
   for text in corpus:
    cmd=[str(a.binary),'tokenize',str(a.model),text]
    # The argv interface cannot represent NUL; test that case through the library suite.
    if '\x00' in text:continue
    if special:cmd+=['--special']
    ids=ast.literal_eval(subprocess.check_output(cmd,text=True))
    if a.tokenizer_oracle:
     oracle=[str(a.tokenizer_oracle),'tokenize',str(a.model),text]
     if special:oracle+=['--special']
     expected=ast.literal_eval(subprocess.run(oracle,check=True,text=True,capture_output=True).stdout)
    else:expected=tk.encode(text,parse_special=special)
    if ids!=expected:raise AssertionError((text,special,ids,expected))
    if tk and tk.decode(ids)!=text:raise AssertionError(('roundtrip',text))
    report['tokenizer'].append({'text':text[:64],'special':special,'tokens':len(ids)})
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('validated',len(formats),'formats and',len(report['tokenizer']),'tokenizer cases')
if __name__=='__main__':main()
