"""Read-only reconciliation of supplied records; does not authenticate live execution."""
import ast, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'results/end_to_end_training_live'
def audit():
    selected={}; attempts=[]; security=[]; hashes=Counter()
    for p in sorted((ROOT/'raw').glob('*/run_manifest.json')):
        m=json.loads(p.read_text()); e=json.loads((p.parent/'evaluation.json').read_text())
        r={'run_id':m['run_id'],'provider':m['model_provider'],'model':m['model_id'],'task':m['task_id'],'condition':m['condition'],'attempt':m['attempt'],'functional':e['functional_status'],'generation':e['details']['generation_status'],'api':e.get('api_conformance_status'),'security':e['security_status']}
        attempts.append(r);key=(r['model'],r['task'],r['condition'])
        if key not in selected or r['attempt']>selected[key]['attempt']:selected[key]=r
        text=(p.parent/'prompt_selected.txt').read_text()
        hashes['normalized_match']+=hashlib.sha256(text.strip().encode()).hexdigest()==m['prompt_sha256']
        for f in e['details'].get('bandit_findings',[]):
            code=(p.parent/'response.txt').read_text();match=re.search(r'```(?:python)?\s*\n(.*?)```',code,re.S);code=match.group(1) if match else code
            n_assert=sum(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(code)))
            security.append({'run_id':r['run_id'],'rule':f['test_id'],'severity':f['severity'],'assert_nodes_in_saved_code':n_assert,'finding_matches_saved_code':n_assert>0 if f['test_id']=='B101' else None})
    conditions={}
    for c in ['baseline','shifted_baseline','shifted_docground']:
        rows=[r for r in selected.values() if r['condition']==c];conditions[c]={'slots':len(rows),'generation':dict(Counter(r['generation'] for r in rows)),'functional':dict(Counter(r['functional'] for r in rows))}
    comparisons=[]
    for a,b in [('baseline','shifted_baseline'),('shifted_baseline','shifted_docground'),('baseline','shifted_docground')]:
        pairs=[];groups=defaultdict(list)
        for (model,task,c),r in selected.items():
            if c!=a:continue
            s=selected.get((model,task,b))
            if s and r['functional'] in ['pass','fail'] and s['functional'] in ['pass','fail'] and r['generation']==s['generation']=='complete':
                delta=int(s['functional']=='pass')-int(r['functional']=='pass');pairs.append((r['functional'],s['functional']));groups[task].append(delta)
        means={t:sum(v)/len(v) for t,v in groups.items()}
        comparisons.append({'before':a,'after':b,'pairs':len(pairs),'transitions':{str(k):v for k,v in Counter(pairs).items()},'pooled_delta_pp':100*sum(sum(v) for v in groups.values())/len(pairs),'task_mean_delta_pp':100*sum(means.values())/len(means),'task_deltas':means,'task_pair_counts':{t:len(v) for t,v in groups.items()}})
    inventory=json.loads((ROOT/'source_inventory.json').read_text());missing=[];changed=[]
    for r in inventory:
        p=ROOT/r['path']
        if not p.exists():missing.append(r['path'])
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:changed.append(r['path'])
    return {'scope':'Arithmetic and static source review only; not independent authentication of provider execution','attempts':len(attempts),'selected_slots':len(selected),'conditions':conditions,'comparisons':comparisons,'prompt_hashes':dict(hashes),'security_review':security,'inventory_missing':missing,'inventory_changed':changed,'provenance_status':'unresolved: missing dataset_provenance.json and fixture wording in chart registry','generation_all_attempts':dict(Counter(r['generation'] for r in attempts))}
if __name__=='__main__':
    out=ROOT/'review/record_reconciliation.json';out.parent.mkdir(exist_ok=True);result=audit();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
