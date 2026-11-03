"""Summarize glm_graph_check.py results without mixing warmup or cached prefill."""
import argparse, json
from pathlib import Path

def rate(rows):
    return 1000 * sum(r['generated'] for r in rows) / sum(r['decode_ms'] for r in rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('results',type=Path);args=ap.parse_args()
    data=json.loads(args.results.read_text(encoding='utf8'));blocks=data['blocks'];modes={}
    for mode in sorted({b['mode'] for b in blocks}):
        group=[b for b in blocks if b['mode']==mode]
        measured=[r for b in group for r in b['rows'] if r['label'].startswith('measure-')]
        cold=[r for b in group for r in b['rows'] if r['label']=='document-cold']
        returns=[r for b in group for r in b['rows'] if r['label']=='document-return']
        modes[mode]={'blocks':len(group),'requests':len(measured),'output_tokens':sum(r['generated'] for r in measured),'decode_tok_s':rate(measured),'code_tok_s':rate([r for r in measured if r['label'].startswith('measure-code-')]),'prose_tok_s':rate([r for r in measured if r['label'].startswith('measure-prose-')]),'post_long_tok_s':rate([r for b in group for r in b['rows'] if r['label'].startswith('post-long-')]),'cold_prefill_tok_s':1000*sum(r['prompt_tokens']-r['reused'] for r in cold)/sum(r['prompt_ms'] for r in cold),'cold_prefill_ms':[r['prompt_ms'] for r in cold],'return_prefill_ms':[r['prompt_ms'] for r in returns],'return_read_tokens':[r['prompt_tokens']-r['reused'] for r in returns]}
    baseline=next(b for b in blocks if b['mode']=='off');ref=[r for r in baseline['rows'] if r['label'].startswith('measure-')]
    parity=[]
    for b in blocks:
        rs=[r for r in b['rows'] if r['label'].startswith('measure-')]
        parity.append({'block':b['block'],'mode':b['mode'],'matches_baseline':sum(x['tokens']==y['tokens'] for x,y in zip(ref,rs)),'requests':len(rs),'decode_tok_s':rate(rs),'graphs':b['graphs'],'primary_slots':b['info']['vram_slots'],'helper_slots':b['info']['helper_slots']})
    checks={'context':all(b['info']['context']=='262144' for b in blocks),'no_mtp':all(b['info']['mtp']=='0' and all(r['drafts_offered']==0 for r in b['rows']) for b in blocks),'gpu_complete':all(b['info']['expert_vram_complete']=='1' for b in blocks),'helper_used':all(any(float(r['stat'].get('helper','0'))>0 for r in b['rows']) for b in blocks),'no_helper_fallback':all(float(r['stat'].get('helper_fb','0'))==0 and r['stat'].get('helper_dead','0')=='0' for b in blocks for r in b['rows']),'no_expert_ssd_reads':all(int(r['stat'].get('expert_ssd_bytes','0'))==0 for b in blocks for r in b['rows']),'exact_return':all(n==1 for m in modes.values() for n in m['return_read_tokens']),'identical_measured_answers_across_all_auto_allocations':all(p['matches_baseline']==p['requests'] for p in parity)}
    summary={'modes':modes,'blocks':parity,'checks':checks,'config_sha256':data['config_sha256']}
    if 'on' in modes and 'off' in modes:
        summary['gain_percent']={k:100*(modes['on'][k]/modes['off'][k]-1) for k in ('decode_tok_s','code_tok_s','prose_tok_s','post_long_tok_s','cold_prefill_tok_s')}
    dest=args.results.parent/'summary.json';dest.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
