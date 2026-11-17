"""Restore the frozen experiment inputs from local or published data archives."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DATA = Path(__file__).resolve().parent
MANIFEST = DATA / "manifests/prepared.json"
DEFAULT_GROUPS = ["hotpot_train", "hotpot_eval", "tokenizers"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Missing file or incorrect size: {path.name}")
    if sha(path) != expected["sha256"]:
        raise ValueError(f"Checksum mismatch: {path.name}")


def destination(data_root, group):
    root = data_root.resolve()
    target = (root / group["destination"]).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("The destination must remain inside the data directory")
    return target


def member_name(name):
    if not name or name in (".", "..") or any(character in name for character in ("/", "\\", ":")):
        raise ValueError("Archive members must be plain filenames")
    return name


def verify_group(data_root, group):
    target = destination(data_root, group)
    for name, expected in group["files"].items():
        member_name(name)
        verify_file(target / name, expected)
    return dict(files=len(group["files"]), bytes=sum(item["bytes"] for item in group["files"].values()))


def download(url, target, expected):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Use a public HTTPS data release URL without embedded credentials")
    if target.exists():
        verify_file(target, expected)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "evidence-utility-data/1"})
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as output:
                temporary = Path(output.name)
                received = 0
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > expected["bytes"]:
                        raise ValueError("Download exceeds the recorded archive size")
                    output.write(chunk)
        verify_file(temporary, expected)
        if target.exists():
            verify_file(target, expected)
        else:
            temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore_group(archive, data_root, group):
    verify_file(archive, group)
    target = destination(data_root, group)
    if target.exists():
        return verify_group(data_root, group)
    target.parent.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(item["bytes"] for item in group["files"].values())
    if shutil.disk_usage(target.parent).free < required_bytes:
        raise ValueError("Not enough free space for the uncompressed inputs")
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".restore-") as folder:
        payload = Path(folder) / "payload"
        payload.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = [member_name(member.name) for member in members]
            if len(names) != len(set(names)) or set(names) != set(group["files"]):
                raise ValueError("Archive members differ from the recorded file list")
            for member in members:
                expected = group["files"][member.name]
                if not member.isfile() or member.size != expected["bytes"]:
                    raise ValueError("Archive member type or size differs")
                with tar.extractfile(member) as source, (payload / member.name).open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                verify_file(payload / member.name, expected)
        payload.rename(target)
    return verify_group(data_root, group)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("restore", "verify"))
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archives-dir", type=Path)
    source.add_argument("--base-url", help="HTTPS URL of the release assets directory")
    args = parser.parse_args()
    if args.action == "restore" and args.archives_dir is None and not args.base_url:
        parser.error("Restoring inputs requires --archives-dir or --base-url")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or any(name not in manifest["groups"] for name in args.groups):
        parser.error("Unknown manifest version or data group")
    result = {}
    try:
        for name in args.groups:
            group = manifest["groups"][name]
            member_name(group["filename"])
            if args.action == "restore":
                archive = (args.archives_dir or args.data_root / "downloads") / group["filename"]
                if args.base_url:
                    print(f"Fetching {group['filename']}", flush=True)
                    download(args.base_url.rstrip("/") + "/" + group["filename"], archive, group)
                result[name] = restore_group(archive, args.data_root, group)
            else:
                result[name] = verify_group(args.data_root, group)
            print(f"Verified {name}", flush=True)
    except (ValueError, OSError, tarfile.TarError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(json.dumps(dict(passed=True, groups=result), indent=2))


if __name__ == "__main__":
    main()
