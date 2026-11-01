"""Read diagnostic traces without loading weights or changing routing.
Global LRU and causal four-expert prediction are screening simulations, not
measured implementations. Demand-only LRU must reproduce real hit/miss counts.
"""
from collections import OrderedDict,Counter,defaultdict
from pathlib import Path
import json,sys

stdout,stderr,output=map(Path,sys.argv[1:])
actual={x['round']:x for x in map(json.loads,[l for l in stdout.read_text().splitlines() if l.startswith('{')])}
events=[];current=None;layers={};commands=defaultdict(list)
for line in stderr.read_text().splitlines():
    if line.startswith('ROUND '):
        parts=line.split();current=int(parts[1]) if parts[2]=='START' else None
    elif line.startswith('ROUTE_LAYER '):
        x=json.loads(line.split(' ',1)[1]);layers[x['shard'],x['offset']]=x['layer']
    elif line.startswith('ROUTE ') and current is not None:
        x=json.loads(line.split(' ',1)[1]);x['round']=current;x['key']=(x['shard'],x['offset']);x['layer']=layers[x['key']];events.append(x)
    elif line.startswith('COMMAND ') and current is not None:commands[current].append(json.loads(line.split(' ',1)[1]))
geometry={x['key']:(x['slots'],x['slot_bytes']) for x in events}
assert len(geometry) in (0,48)
budget=sum(slots*size for slots,size in geometry.values())
plain=defaultdict(OrderedDict);global_lru=OrderedDict();global_bytes=0
predicted=defaultdict(OrderedDict);training=defaultdict(Counter);previous=None;prefetched=set()
metrics=defaultdict(Counter);per_layer=defaultdict(Counter)
for rnd in actual:metrics[rnd]
def demand(cache,expert,capacity):
    hit=expert in cache
    if hit:cache.move_to_end(expert)
    else:
        if len(cache)==capacity:cache.popitem(last=False)
        cache[expert]=True
    return hit
for event in events:
    rnd=event['round'];key=event['key'];layer=event['layer'];slots,size=geometry[key];ids=[x[0] for x in event['groups']];m=metrics[rnd]
    m['groups']+=1;m['expert_assignments']+=sum(x[1] for x in event['groups']);m['unique_expert_groups']+=len(ids)
    condition=None
    if event['count']==1 and previous is not None and previous['count']==1 and previous['layer']==layer-1 and previous['round']==rnd:
        condition=(layer,previous['groups'][0][0])
        guesses=sorted(training[condition],key=lambda i:(-training[condition][i],i))[:4]
        for expert in guesses:
            if expert in predicted[key]:continue
            if len(predicted[key])==slots:
                victim,_=predicted[key].popitem(last=False);m['prefetch_evictions']+=1
                if (key,victim) in prefetched:prefetched.remove((key,victim));m['prefetch_evicted_unused']+=1
            predicted[key][expert]=True;prefetched.add((key,expert));m['prefetch_reads']+=1;m['prefetch_bytes']+=size
    for expert in ids:
        hit=demand(plain[key],expert,slots);m['plain_hits' if hit else 'plain_misses']+=1
        if not hit:m['plain_read_bytes']+=size
        per_layer[rnd,layer]['hits' if hit else 'misses']+=1
        global_key=(key,expert)
        if global_key in global_lru:global_lru.move_to_end(global_key);m['global_hits']+=1
        else:
            m['global_misses']+=1;m['global_read_bytes']+=size
            while global_bytes+size>budget:
                _,removed=global_lru.popitem(last=False);global_bytes-=removed
            global_lru[global_key]=size;global_bytes+=size
        if expert in predicted[key]:
            predicted[key].move_to_end(expert);m['predicted_hits']+=1
            if (key,expert) in prefetched:prefetched.remove((key,expert));m['prefetch_used']+=1
        else:
            m['predicted_misses']+=1;m['predicted_demand_bytes']+=size
            if len(predicted[key])==slots:
                victim,_=predicted[key].popitem(last=False)
                if (key,victim) in prefetched:prefetched.remove((key,victim));m['prefetch_evicted_unused']+=1
            predicted[key][expert]=True
    if condition is not None:training[condition].update(ids)
    previous=event
for rnd,m in metrics.items():
    measured=actual[rnd]
    assert (m['plain_hits'],m['plain_misses'])==(measured['cache_hits'],measured['cache_misses']),('route trace does not reproduce actual cache accounting',rnd,m,measured)
    m['measured_read_s']=measured['read_s'];m['measured_total_s']=measured['total_s']
    m['ideal_zero_read_seconds_upper_bound']=min(measured['read_s'],measured['total_s'])
    m['ideal_zero_read_fraction_upper_bound']=min(1.,measured['read_s']/measured['total_s'])
    cs=commands[rnd];m['command_count']=len(cs);m['encoder_count']=sum(c['encoders'] for c in cs)
    m['cpu_encode_s']=sum(c['encode_s'] for c in cs);m['cpu_blocking_wait_s']=sum(c['blocking_wait_s'] for c in cs)
    timed=[c for c in cs if c['encoders'] and c['gpu_end']>c['gpu_start']>0]
    m['gpu_timed_commands']=len(timed)
    m['gpu_busy_s']=sum(c['gpu_end']-c['gpu_start'] for c in timed)
    m['gpu_gap_s']=sum(max(0.,b['gpu_start']-a['gpu_end']) for a,b in zip(timed,timed[1:]))
result={'diagnostic_only':True,'slot_budget_bytes':budget,'rounds':dict(metrics),
        'per_layer':[{'round':rnd,'layer':layer,**dict(counts)} for (rnd,layer),counts in sorted(per_layer.items())],
        'unconsumed_prefetches_at_end':len(prefetched),
        'notes':['Runtime kernels and normal command boundaries retained; added CPU diagnostics have overhead. These timings are not formal performance samples.',
                 'Read time already overlaps GPU work; the zero-read bound intentionally overestimates any possible speedup. Never add read, GPU and CPU wait durations.',
                 'Global LRU simulation allows a shared byte pool without pinning constraints; this optimistic simplification is not a deployed allocator.',
                 'Predictor uses only earlier decode observations: current layer indexed by the previous layer lowest selected expert ID (trace groups are sorted by expert ID, not route rank), top four historical conditional frequencies. Current demand IDs are used only after prediction, then training updates.',
                 'Prefetch read/eviction/unused counts measure a causal trace simulation, not achieved latency or physical SSD traffic.']}
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['rounds'],indent=2))
