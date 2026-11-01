"""Select weights for the single Maya backend and shared server in this checkout."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
PROFILES={"deepseek": "maya-ds4.json", "glm": "maya-glm.json", "minimax": "maya-minimax.json"}
def available(root=ROOT):
    result=[]
    preferred=[root/n for n in PROFILES.values()]
    paths=preferred+[p for p in sorted(root.glob("maya-*.json")) if p not in preferred]
    for p in paths:
        if not p.is_file():continue
        try:c=json.loads(p.read_text(encoding="utf-8-sig"))
        except (ValueError,OSError):continue
        if not c.get("exe") or not isinstance(c.get("args"),list):continue
        args=c["args"];model=c.get("model_name", "").lower()
        family=("deepseek" if "deepseek" in model or p.name=="maya-ds4.json" else
                "minimax" if "minimax" in model else "glm" if "glm" in model or "--glm-pack" in args else None)
        if not family:continue
        c["exe"]=str(root/"build/maya-backend.exe");c["cwd"]=str(root)
        if "--family" not in args:c["args"]=["--family",family]+args
        result.append((family,p,c))
    return result

def main(argv=None):
    ap=argparse.ArgumentParser(description="Maya: one backend, choose the model weights")
    ap.add_argument("--model",choices=list(PROFILES))
    ap.add_argument("--list",action="store_true")
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--port",type=int)
    a=ap.parse_args(argv);profiles=available()
    if not profiles:ap.error("No installed model profiles: run START-MAYA.bat --setup first")
    if a.list:
        for family,p,c in profiles:print(f"{family}: {c.get('model_name',family)} ({p.name})")
        return 0
    selected=a.model;record=None
    if selected is None:
        print("\nMaya - backend unico\nSeleziona i pesi da caricare:\n")
        for i,(family,p,c) in enumerate(profiles,1):print(f"  {i}. {c.get('model_name',family)}")
        print("  0. Esci\n")
        try:choice=input("Scelta: ").strip()
        except (EOFError,KeyboardInterrupt):return 0
        if choice=="0":return 0
        if not choice.isdigit() or not 1<=int(choice)<=len(profiles):ap.error("Scelta non valida")
        record=profiles[int(choice)-1];selected=record[0]
    record=record or next((x for x in profiles if x[0]==selected),None)
    if record is None:ap.error("Model profile unavailable")
    _,p,c=record
    command=[sys.executable,str(ROOT/"maya.py"),"--config",str(p)]
    if a.port is not None:command.extend(["--port",str(a.port)])
    if a.dry_run:
        print(json.dumps({"config":str(p),"backend":c["exe"],"backend_args":c["args"],"command":command},ensure_ascii=False,indent=2));return 0
    if not Path(c["exe"]).is_file():ap.error("Build the single backend first with build-unified.bat")
    p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    os.chdir(ROOT);os.execv(sys.executable,command)
if __name__=="__main__":raise SystemExit(main())
