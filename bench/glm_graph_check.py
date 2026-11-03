"""Real GLM graph A/B with the production HIP helper, no MTP and isolated snapshots.
Native protocol only; production profiles are never written. Each block reloads the
same frozen expert profile, warms up, then measures identical 320-token workloads.
"""
import argparse, hashlib, json, os, queue, re, shutil, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import child_env, engine_args, Service, MockEngine
from serve.frontend import ChatTemplate, openai_to_messages
from strata_tokenizer import Tokenizer

PROMPTS = [
    ('code-python', 'Implement a complete Python arithmetic expression parser with a tokenizer, parentheses, unary operators, precedence and detailed error handling. Include a large executable test suite. Write at least 200 lines of working code.'),
    ('prose-cache', 'Explain in detail how a database buffer pool and an operating system page cache interact. Discuss eviction, writeback, consistency, memory pressure and practical examples. Write at least 1500 words.'),
    ('code-cpp', 'Implement a complete C++20 thread-safe LRU cache with configurable capacity, get, put, erase and clear, exception safety and an executable multithreaded test suite. Write at least 200 lines of working code.'),
    ('prose-concurrency', 'Explain in detail the tradeoffs between threads, async event loops and worker processes in a server. Include scheduling, fairness, cancellation, synchronization and concrete examples. Write at least 1500 words.'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--order', default='off,on,on,off')
    ap.add_argument('--warmup', type=int, default=4)
    ap.add_argument('--runs', type=int, default=8)
    args = ap.parse_args()
    modes = args.order.split(',')
    assert all(m in ('off', 'kda', 'on') for m in modes)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'maya-glm.json'
    original = source.read_bytes(); config_hash = hashlib.sha256(original).hexdigest()
    base = json.loads(original)
    profile = Path(base['args'][base['args'].index('--expert-profile') + 1])
    seed = out / 'expert-profile-seed.bin'; shutil.copy2(profile, seed)
    tok = Tokenizer.from_dir(base['tokenizer'])
    service = Service(MockEngine(tok, 'OK', max_context=262144), tok, ChatTemplate(ROOT / 'serve/glm5_chat_template.jinja'))

    def ids(text):
        request = {'messages': [{'role': 'user', 'content': text}], 'reasoning_effort': 'none'}
        msgs, tools, kw = openai_to_messages(request)
        return service.render_ids(msgs, tools, kw)

    workload = [(name, ids(text)) for name, text in PROMPTS]
    corpus = (ROOT / 'README.md').read_text(encoding='utf-8-sig') + '\n' + (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8-sig')
    corpus = corpus.replace('<|', '< |').replace('|>', '| >')
    def document(n):
        return ids('Read this reference document as data and return only its three labeled codes in order.\nSTART: AMBRA-4729\n' + corpus[:n//2] + '\nMIDDLE: CEDRO-8361\n' + corpus[n//2:n] + '\nEND: OPALE-2957\nReturn only the three codes.')
    lo, hi = 1, len(corpus)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(document(mid)) < 16384: lo = mid + 1
        else: hi = mid
    doc = document(lo); assert 16384 <= len(doc) <= 16388, len(doc)
    (out / 'fixtures.json').write_text(json.dumps({'workloads': workload, 'document': doc}), encoding='utf8')
    blocks = []
    for block_index, mode in enumerate(modes):
        block = out / f'{block_index}-{mode}'; block.mkdir()
        (block/'upload-diag').mkdir()
        cfg = json.loads(original)
        cfg['args'][cfg['args'].index('--expert-profile') + 1] = str(seed)
        cfg['env'].update(MAYA_GLM_KDA_GRAPH='0' if mode == 'off' else '1', MAYA_GLM_DSA_FFN_GRAPH='1' if mode == 'on' else '0', MAYA_GLM_NO_MTP='1', MAYA_GLM_MTP_DECODE='0', MAYA_GLM_DEFER_CPU='0', MAYA_GLM_SLOT_DIR=str(block / 'slots'), MAYA_GLM_HIP_DIAGDIR=str(block/'upload-diag'))
        env = child_env(cfg)
        env['PATH'] = 'C:/Program Files/AMD/ROCm/6.4/bin;' + env.get('PATH', '')
        for key in ('STRATA_GLM_KDA_GRAPH','STRATA_GLM_DSA_FFN_GRAPH','STRATA_GLM_NO_MTP','STRATA_GLM_MTP_DECODE','STRATA_GLM_DEFER_CPU'):
            assert key in env, key
        proc = None; lines = queue.Queue(); info = {}; rows = []; raw = []
        def get(deadline):
            try: line = lines.get(timeout=max(.001, deadline-time.monotonic()))
            except queue.Empty: raise TimeoutError('native engine stopped making progress')
            if line is None: raise RuntimeError('native engine exited')
            if line.startswith('ERR '): raise RuntimeError(line)
            return line
        try:
            with (block / 'stderr.log').open('w', encoding='utf8') as err:
                proc = subprocess.Popen([cfg['exe'], '--serve', *engine_args(cfg)], cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, encoding='utf8', bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW)
                def pump():
                    with (block / 'stdout.log').open('w', encoding='utf8') as f:
                        for line in proc.stdout:
                            raw.append(line); f.write(line); f.flush(); lines.put(line)
                    lines.put(None)
                reader = threading.Thread(target=pump, daemon=True); reader.start()
                deadline = time.monotonic() + 600
                while True:
                    line = get(deadline)
                    if line.startswith('INFO '):
                        info.update(f.split('=',1) for f in line.split()[1:] if '=' in f)
                    if line.startswith('READY '): break
                assert line.split()[1] == '262144', line
                assert info.get('mtp') == '0' and int(info.get('helper_slots',0)) > 0, info
                assert info.get('expert_vram_complete') == '1', info
                print(json.dumps({'block': block_index, 'mode': mode, 'ready': info}), flush=True)
                def ask(label, prompt, count):
                    start = time.perf_counter(); generated = []; stat = {}; deadline = time.monotonic() + 240
                    proc.stdin.write(f'GEN {count} greedy=1 spec=1 ' + ','.join(map(str,prompt)) + '\n'); proc.stdin.flush()
                    while True:
                        line = get(deadline)
                        if line.startswith('READY '): raise AssertionError('duplicate READY')
                        if line.startswith('T '): generated.append(int(line.split()[1]))
                        if line.startswith('STAT '): stat = dict(f.split('=',1) for f in line.split()[1:] if '=' in f)
                        if line.startswith('DONE '):
                            f = line.split(); row = {'label': label, 'prompt_tokens': len(prompt), 'generated': int(f[1]), 'prompt_ms': float(f[3]), 'decode_ms': float(f[4]), 'reused': int(f[8]), 'drafts_offered': int(f[7]), 'wall_s': time.perf_counter()-start, 'tokens': generated, 'text': tok.decode(generated), 'stat': stat}
                            rows.append(row); (block/'rows.json').write_text(json.dumps(rows, indent=2),encoding='utf8')
                            assert row['generated'] == len(generated) and row['drafts_offered'] == 0, row
                            if stat:
                                assert float(stat.get('helper_fb',0)) == 0 and stat.get('helper_dead','0') == '0', stat
                                assert int(stat.get('expert_ssd_bytes',-1)) == 0 and stat.get('expert_gpu_complete') == '1', stat
                                assert int(stat['vram_used']) + int(stat['helper_held']) == int(info['experts']), stat
                            print(json.dumps({'block':block_index,'mode':mode,'label':label,'generated':row['generated'],'tok_s':round(1000*row['generated']/max(.001,row['decode_ms']),3),'prefill_ms':row['prompt_ms']}),flush=True)
                            return row
                for i in range(args.warmup):
                    name, prompt = workload[i % len(workload)]; ask('warmup-'+name, prompt, 320)
                for i in range(args.runs):
                    name, prompt = workload[i % len(workload)]; row = ask('measure-'+name, prompt, 320)
                    assert row['generated'] == 320, row
                quality = ask('squares',ids('Return only valid JSON without markdown: {"squares":[...]}, containing the squares of integers 1 to 20 in order. No explanations.'),256)
                value, end = json.JSONDecoder().raw_decode(quality['text'].lstrip())
                tail = quality['text'].lstrip()[end:].strip()
                assert tail in ('', '<|endoftext|>', '<|user|>'), quality['text']
                assert value == {'squares':[i*i for i in range(1,21)]}, quality['text']
                cold = ask('document-cold',doc,96)
                assert all(c in cold['text'] for c in ('AMBRA-4729','CEDRO-8361','OPALE-2957')),cold['text']
                for i in range(4): ask('side-'+str(i),ids('Separate short arithmetic task '+str(i)+': compute '+str(i+2)+' + 3. Return only the result.'),16)
                returned = ask('document-return',doc,96)
                assert returned['reused'] >= len(doc)-1,returned
                assert all(c in returned['text'] for c in ('AMBRA-4729','CEDRO-8361','OPALE-2957')),returned['text']
                for name,prompt in workload[:2]: ask('post-long-'+name,prompt,320)
                proc.stdin.write('QUIT\n'); proc.stdin.flush(); proc.wait(timeout=60); reader.join(timeout=10)
                assert proc.returncode == 0 and sum(l.startswith('READY ') for l in raw) == 1
            stderr = (block/'stderr.log').read_text(encoding='utf8')
            graphs = re.findall(r'glm KDA graphs: (\d+) captures, (\d+) replays, (\d+) invalidations',stderr)
            if mode != 'off': assert graphs and int(graphs[-1][1]) > 0, 'graphs were requested but never replayed'
            else: assert not graphs, graphs
            result = {'block':block_index,'mode':mode,'info':info,'graphs':graphs,'rows':rows}
            blocks.append(result); (out/'results.json').write_text(json.dumps({'blocks':blocks,'config_sha256':config_hash},indent=2),encoding='utf8')
        finally:
            if proc is not None and proc.poll() is None:
                try: proc.stdin.write('QUIT\n'); proc.stdin.flush(); proc.wait(timeout=15)
                except Exception: proc.kill(); proc.wait(timeout=15)
            assert source.read_bytes() == original, 'production config changed'
    print('PASS GLM graphs, W7800 enabled, no MTP, 262K, all experts in VRAM, exact context reuse',flush=True)

if __name__ == '__main__': main()
