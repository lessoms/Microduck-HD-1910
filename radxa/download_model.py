"""Fetch and verify the release model without Git LFS or third-party packages."""
import argparse,hashlib,json,os,tempfile,urllib.request
from pathlib import Path

def verify(path,manifest):
    if not path.is_file() or path.stat().st_size!=manifest['size']:return False
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()==manifest['sha256']

def prepare(root):
    manifest=json.loads((root/'model_manifest.json').read_text());dest=root/manifest['filename']
    if verify(dest,manifest):print('Model checksum OK');return dest
    if dest.exists():raise RuntimeError('Existing model checksum differs; move it aside before downloading')
    temporary=root/(manifest['filename']+'.download')
    try:
        request=urllib.request.Request(manifest['url'],headers={'User-Agent':'Microduck-HD-1910-Radxa/1.0'})
        digest=hashlib.sha256();total=0
        with urllib.request.urlopen(request,timeout=90) as response,temporary.open('wb') as output:
            while True:
                block=response.read(1024*1024)
                if not block:break
                total+=len(block)
                if total>manifest['size']:raise RuntimeError('Download larger than expected')
                digest.update(block);output.write(block)
        if total!=manifest['size'] or digest.hexdigest()!=manifest['sha256']:raise RuntimeError('Model size or SHA-256 mismatch')
        temporary.replace(dest);print('Model downloaded and SHA-256 verified')
    finally:
        temporary.unlink(missing_ok=True)
    return dest
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();prepare(a.directory)
