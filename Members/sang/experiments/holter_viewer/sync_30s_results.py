"""Poll a known Furiosa run; copy only its score CSV and status to local storage."""

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--key", type=Path, default=Path("/mnt/c/Users/Admin/.ssh/id_ed25519_std3"))
    args = p.parse_args()
    if not re.fullmatch(r"segment30_[0-9TZ]+_[a-f0-9]{6}", args.name):
        raise ValueError("Invalid run name")
    if not (args.local / "selection.json").is_file():
        raise ValueError("Missing local selection")
    root = "/root/.cache/std3_holter/llm_compare/" + args.name
    code = "import json,sys,pathlib;p=pathlib.Path(sys.argv[1]);m=p/'metadata.json';s=p/'model_scores.csv';print(json.dumps({'metadata':json.loads(m.read_text()) if m.exists() else None,'csv':s.read_text() if s.exists() else None}))"
    command = " ".join(
        shlex.quote(x) for x in ["/root/std3/furiosa_holter_venv/bin/python", "-c", code, root]
    )
    with tempfile.TemporaryDirectory(prefix="furiosa-sync-") as d:
        key = Path(d) / "key"
        key.write_bytes(args.key.read_bytes())
        key.chmod(0o600)
        while True:
            try:
                result = subprocess.run(
                    [
                        "ssh",
                        "-i",
                        str(key),
                        "-p",
                        "36040",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "IdentitiesOnly=yes",
                        "-o",
                        "ConnectTimeout=8",
                        "std3@eval.nxt.furiosa.ai",
                        command,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=25,
                    check=True,
                )
                obj = json.loads(result.stdout)
                meta = obj["metadata"]
                if obj["csv"] is not None:
                    temp = args.local / ".scores.tmp"
                    temp.write_text(obj["csv"])
                    temp.chmod(0o600)
                    os.replace(temp, args.local / "model_scores.csv")
                if meta is not None:
                    temp = args.local / ".meta.tmp"
                    temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                    temp.chmod(0o600)
                    os.replace(temp, args.local / "metadata.json")
                    if meta["status"] in ("COMPLETE", "PARTIAL", "FAILED"):
                        print("SYNC_FINISHED", meta["status"], flush=True)
                        break
            except (OSError, subprocess.SubprocessError, ValueError, KeyError):
                print("SYNC_RETRY", flush=True)
            time.sleep(15)


if __name__ == "__main__":
    main()
