import argparse,json,sys
from pathlib import Path
import numpy as np,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretrain/data_process'))
from pack_motion_library import subset_library
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
for split,count in [('train',256),('eval',32)]:
    assert (a.source/split/'COMPLETE').is_file()
    index=json.loads((a.source/split/'index.json').read_text())
    ids=np.random.default_rng(20261029).choice(len(index['motion_names']),min(count,len(index['motion_names'])),replace=False)
    manifest=a.out/(split+'.yaml');manifest.write_text(yaml.safe_dump({index['motion_names'][i]:'' for i in sorted(ids)}))
    subset_library(a.source/split,manifest,a.out/split)
