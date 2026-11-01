"""The BAT selects weights in this checkout, always through one backend."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("launcher",ROOT/"tools/launch_models.py")
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
class UnifiedLauncher(unittest.TestCase):
 @unittest.skipUnless(all((ROOT/n).is_file() for n in launcher.PROFILES.values()), "local installed profiles required")
 def test_local_profiles_use_same_backend_and_preserve_tuned_options(self):
  profiles=launcher.available();self.assertEqual({p[0] for p in profiles},{"deepseek","glm","minimax"})
  exes={Path(p[2]["exe"]).resolve() for p in profiles};self.assertEqual(exes,{ROOT/"build/maya-backend.exe"})
  for family,path,cfg in profiles:
   args=cfg["args"];self.assertEqual(args[:2],["--family",family]);self.assertEqual(Path(cfg["cwd"]),ROOT)
   ctx=args[args.index("--max-context")+1];self.assertEqual(ctx,"131072" if family=="minimax" else "262144")
   if family in ("deepseek","minimax"):self.assertEqual(args[args.index("--dspark")+1],"on")
   if family=="deepseek":self.assertEqual(args[args.index("--mmap-experts")+1],"off")
   if family=="minimax":self.assertEqual(args[args.index("--kv-cache")+1],"int8")
 def test_discovery_migrates_real_legacy_profiles_without_launching_old_engines(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/"maya-ds4.json").write_text(json.dumps({"exe":"build/maya-ds4.exe","args":["--pack","weights"],"model_name":"deepseek-v4.1"}))
   profiles=launcher.available(root)
   self.assertEqual(Path(profiles[0][2]["exe"]),root/"build/maya-backend.exe")
   self.assertEqual(profiles[0][2]["args"][:2],["--family","deepseek"])
if __name__=="__main__":unittest.main()
