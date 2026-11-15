"""Verify packaged assets and actor inference without a GPU."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import microduck_sm.policy as policy_module
import mjlab_microduck
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
assert Path(policy_module.__file__).resolve().is_relative_to(ROOT)
assert Path(mjlab_microduck.__file__).resolve().is_relative_to(ROOT/'vendor')
manifest=json.loads((ROOT/'models/manifest.json').read_text())
reports=[]
for entry in manifest['models']:
 path=ROOT/entry['model'];reference=ROOT/entry['reference']
 assert sha(path)==entry['model_sha256']
 assert sha(reference)==entry['reference_sha256']
 saved=torch.load(path,map_location='cpu',weights_only=True)
 cfg=saved['collection_config']
 assert not Path(cfg['reference']).is_absolute()
 assert cfg['reference']==entry['reference']
 with np.load(reference) as data:
  assert len(data['root_pose'])>cfg['reference_end']
 model=policy_module.OldAlignedPolicy().eval()
 model.load_state_dict(saved['models']['policy'])
 with torch.no_grad():
  output=model(torch.zeros(2,480),torch.zeros(2,440))
 assert output.shape==(2,14) and torch.isfinite(output).all()
 reports.append(dict(name=entry['name'],sha256_verified=True,output_shape=list(output.shape),finite=True))
result=dict(passed=True,models=reports,policy_import=str(Path(policy_module.__file__).relative_to(ROOT)),robot_import=str(Path(mjlab_microduck.__file__).relative_to(ROOT)))
(ROOT/'outputs').mkdir(exist_ok=True)
(ROOT/'outputs/package_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
