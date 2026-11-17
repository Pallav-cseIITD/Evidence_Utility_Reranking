"""Data restoration checks using small local archives and mocked downloads."""

import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from data.prepare import destination, download, member_name, restore_group, sha, verify_group


def entry(content):
    return dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def archive_fixture(root, members=None, expected=None):
    members = members if members is not None else [("records.json", b"[1,2,3]\n", False)]
    archive = root / "example.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, content, symlink in members:
            info = tarfile.TarInfo(name)
            if symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = "other.txt"
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    group = dict(filename=archive.name, bytes=archive.stat().st_size, sha256=sha(archive),
                 destination="prepared/example", files=expected or {"records.json": entry(b"[1,2,3]\n")})
    return archive, group


class RestoreTests(unittest.TestCase):
    def test_restore_and_reuse_preserve_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, group = archive_fixture(root)
            data = root / "data"
            first = restore_group(archive, data, group)
            self.assertEqual((data / "prepared/example/records.json").read_bytes(), b"[1,2,3]\n")
            self.assertEqual(first, verify_group(data, group))
            self.assertEqual(first, restore_group(archive, data, group))

    def test_modified_existing_inputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, group = archive_fixture(root)
            restore_group(archive, root / "data", group)
            output = root / "data/prepared/example/records.json"
            output.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                restore_group(archive, root / "data", group)
            self.assertEqual(output.read_bytes(), b"changed")

    def test_bad_archive_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, group = archive_fixture(root)
            group["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                restore_group(archive, root / "data", group)
            self.assertFalse((root / "data/prepared/example").exists())

    def test_bad_member_checksum_leaves_no_completed_group(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, group = archive_fixture(root)
            group["files"]["records.json"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                restore_group(archive, root / "data", group)
            self.assertFalse((root / "data/prepared/example").exists())
            self.assertEqual(list((root / "data/prepared").iterdir()), [])

    def test_duplicate_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            members = [("records.json", b"[1,2,3]\n", False)] * 2
            archive, group = archive_fixture(root, members)
            with self.assertRaisesRegex(ValueError, "recorded file list"):
                restore_group(archive, root / "data", group)

    def test_unlisted_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, group = archive_fixture(root, [("other.json", b"[1,2,3]\n", False)])
            with self.assertRaisesRegex(ValueError, "recorded file list"):
                restore_group(archive, root / "data", group)

    def test_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, group = archive_fixture(root, [("records.json", b"", True)], {"records.json": entry(b"")})
            with self.assertRaisesRegex(ValueError, "type or size"):
                restore_group(archive, root / "data", group)

    def test_paths_must_remain_inside_the_data_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                destination(root / "data", {"destination": "../outside"})
            for name in ("../outside", "/absolute", "dir/file", "dir\\file", "C:drive", ".."):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    member_name(name)


class DownloadTests(unittest.TestCase):
    def test_verified_download_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "cache/archive.tar.gz"
            content = b"example archive bytes"
            with patch("data.prepare.urlopen", return_value=io.BytesIO(content)) as request:
                download("https://example.org/archive.tar.gz", target, entry(content))
                self.assertEqual(target.read_bytes(), content)
                self.assertEqual(request.call_count, 1)
            with patch("data.prepare.urlopen") as request:
                download("https://example.org/archive.tar.gz", target, entry(content))
                request.assert_not_called()

    def test_oversized_download_is_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "archive.tar.gz"
            with patch("data.prepare.urlopen", return_value=io.BytesIO(b"too many bytes")):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    download("https://example.org/archive.tar.gz", target, entry(b"small"))
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_bad_download_checksum_is_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "archive.tar.gz"
            expected = entry(b"same size")
            with patch("data.prepare.urlopen", return_value=io.BytesIO(b"not right")):
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    download("https://example.org/archive.tar.gz", target, expected)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_download_requires_public_https(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "archive.tar.gz"
            for url in ("file:///tmp/example", "http://example.org/archive", "https://user:secret@example.org/archive"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    download(url, target, entry(b"content"))


if __name__ == "__main__":
    unittest.main()
