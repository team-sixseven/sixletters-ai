"""Reconstruct only selected 128-window inputs in sparse arrays; do not use as full ECG."""
import csv,hashlib,json,os,tarfile,io
from pathlib import Path
from collections import defaultdict
import numpy as np
base=Path('/workspace/dl_six'); src=Path('/tmp/dl_six_snapshot.tar'); dest=Path('/tmp/dl_six_windows');dest.mkdir(parents=True,exist_ok=True)
if any(dest.iterdir()):raise RuntimeError('snapshot directory must be empty')
with tarfile.open(src) as tar:
 m=json.load(tar.extractfile('manifest.json')); expected=m['records']; selected=defaultdict(list)
 quality=base/'snapshot/Holter/segment_rpeaks.csv'
 with quality.open(newline='') as f:
  for row in csv.DictReader(f):
   p=row['file'].removesuffix('.npy')
   if p in expected and row['exclude'].lower()=='false':selected[p].append(int(row['window_idx']))
 for n,(p,entry) in enumerate(expected.items(),1):
  valid=np.array(sorted(selected[p]),dtype=np.int64)
  picked=valid[np.linspace(0,len(valid)-1,min(128,len(valid)),dtype=np.int64)]
  if picked.tolist()!=entry['indices']:raise RuntimeError('selected indices mismatch')
  with tar.extractfile(f'segments/{p}.npy') as stream:windows=np.load(io.BytesIO(stream.read()),allow_pickle=False)
  if hashlib.sha256(windows.tobytes()).hexdigest()!=entry['sha256']:raise RuntimeError('segment hash mismatch')
  a=np.lib.format.open_memmap(dest/f'{p}.npy',mode='w+',dtype=np.float16,shape=tuple(entry['shape']))
  for i,j in enumerate(picked):a[j*6000:(j+1)*6000]=windows[i]
  a.flush();del a
  a=np.load(dest/f'{p}.npy',mmap_mode='r',allow_pickle=False)
  if hashlib.sha256(np.stack([a[j*6000:(j+1)*6000] for j in picked]).tobytes()).hexdigest()!=entry['sha256']:raise RuntimeError('reconstructed selected samples mismatch')
  if n%50==0:print('verified',n,'/',len(expected),flush=True)
 (base/'snapshot/README.txt').write_text(m['method']+'\nDo not use for any other sampling or as original data.\n')
 (base/'snapshot/manifest.json').write_text(json.dumps(m,separators=(',',':')))
(base/'snapshot/Holter/windows').symlink_to(dest, target_is_directory=True)
print('SNAPSHOT_VERIFIED',len(expected),flush=True)
