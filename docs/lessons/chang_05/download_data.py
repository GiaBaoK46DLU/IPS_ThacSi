"""Tải đúng hai CSV từ ZIP UCI, hoặc đối chiếu repo mà không ghi raw."""
import argparse
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from data_utils import RAW, ROOT, json_bytes, sha256

URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
EXPECTED = {
    "trainingData.csv": "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168",
    "validationData.csv": "5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=RAW)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with urlopen(URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        contents = {name: archive.read("UJIndoorLoc/" + name) for name in EXPECTED}
    # Chỉ chọn hai member đã biết, không extractall đường dẫn tùy ý từ ZIP.
    for name, content in contents.items():
        if hashlib.sha256(content).hexdigest() != EXPECTED[name]:
            raise ValueError(f"Nguồn UCI đã đổi: {name}. Kiểm tra phiên bản trước khi dùng.")
        destination = args.dest / name
        if destination.exists() and sha256(destination) != EXPECTED[name]:
            raise ValueError(f"Tệp local khác UCI: {destination}; không ghi đè.")
        if args.verify_only and not destination.exists():
            raise FileNotFoundError(destination)
    if not args.verify_only:
        args.dest.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            if not (args.dest / name).exists():
                (args.dest / name).write_bytes(content)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": "https://archive.ics.uci.edu/dataset/310/ujiindoorloc",
        "archive_url": URL, "archive_bytes": len(archive_bytes),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "files": {name: {"member": "UJIndoorLoc/" + name, "sha256": EXPECTED[name],
                         "bytes": len(content), "local_matches_official": True}
                  for name, content in contents.items()},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(json_bytes(report))
    print("PASS: hai CSV trùng từng byte với bản UCI đã xác minh.")


if __name__ == "__main__":
    main()
