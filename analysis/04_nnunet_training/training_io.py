"""Portable file metadata for the official training instrumentation."""

import hashlib
import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PREPROCESSED_DATASET = Path("preprocessed") / "Dataset701_ACDCDevelopment"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_development_case(identifier):
    match = re.fullmatch(r"ACDC_(\d{3})_(ED|ES)", identifier)
    if match is None or not 1 <= int(match[1]) <= 100:
        raise ValueError(f"Not a development case: {identifier}")
    return int(match[1]), match[2]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def relative_path(path):
    return Path(path).as_posix()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    source = Path(path)
    return {"path": source.as_posix(), "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size}


def assert_output_path(path):
    return Path(path)


def set_nnunet_environment():
    os.environ["nnUNet_compile"] = "false"
    os.environ["nnUNet_n_proc_DA"] = "8"
