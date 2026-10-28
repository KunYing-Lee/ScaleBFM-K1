"""Concatenate complete libraries byte-for-byte, preserving source clip membership."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
for split in ['train','eval']:
    sources=[s/split for s in a.sources];indices=[json.loads((s/'index.json').read_text()) for s in sources]
    assert all(not (s/'INCOMPLETE').exists() and (s/'COMPLETE').exists() for s in sources)
    first=indices[0]
    assert all(all(x[k]==first[k] for k in ['fps','joint_names','body_names']) for x in indices)
    index={k:first[k] for k in ['fps','joint_names','body_names']}
    index.update(motion_names=[n for x in indices for n in x['motion_names']],lengths=[n for x in indices for n in x['lengths']],source_libraries=list(map(str,sources)))
    assert len(set(index['motion_names']))==len(index['motion_names'])
    out=a.out/split;out.mkdir();(out/'INCOMPLETE').touch();(out/'index.json').write_text(json.dumps(index))
    for field in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
        arrays=[np.load(s/f'{field}.npy',mmap_mode='r') for s in sources]
        dest=np.lib.format.open_memmap(out/f'{field}.npy',mode='w+',dtype='float32',shape=(sum(index['lengths']),*arrays[0].shape[1:]))
        offset=0
        for arr in arrays:
            for start in range(0,len(arr),65536):
                chunk=arr[start:start+65536];dest[offset+start:offset+start+len(chunk)]=chunk
                assert np.array_equal(dest[offset+start:offset+start+len(chunk)],chunk)
            offset+=len(arr)
        dest.flush();del dest
    (out/'INCOMPLETE').unlink();(out/'COMPLETE').write_text('All input frames copied and compared exactly.\n')
    print('MERGE_PASS',split,len(index['lengths']),sum(index['lengths']),flush=True)
