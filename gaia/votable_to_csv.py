"""turn a gaia archive VOTable download into the csv that data.py reads"""

from __future__ import annotations

import base64
import csv
import gzip
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # the pipeline modules are not a package

from config import CONFIG
from data import EXPECTED_COLUMNS

# big-endian struct code and byte width per VOTable datatype
NUMERIC = {
    "short": (">h", 2), "int": (">i", 4), "long": (">q", 8),
    "float": (">f", 4), "double": (">d", 8), "unsignedByte": (">B", 1),
}


def _tag(element) -> str:
    """local name without the VOTable xml namespace"""
    return element.tag.rsplit("}", 1)[-1]


def _first(root, name):
    return next((e for e in root.iter() if _tag(e) == name), None)


def _read_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def _format(value, datatype: str) -> str:
    """render one value the way the archive's own csv export does"""
    if value is None:
        return ""
    if datatype == "float":
        return str(np.float32(value))  # shortest text that round-trips as float32
    if datatype == "double":
        return repr(float(value))
    return str(value)


def _decode_binary2(payload: str, datatypes: list[str]) -> list[list]:
    """
    BINARY2 packs every row as a null-flag bitmask followed by the raw big-endian values
    (BINARY1 has no mask, so the offsets would all be one byte off)
    """
    blob = base64.b64decode(payload)
    mask_width = (len(datatypes) + 7) // 8
    rows, at = [], 0

    while at < len(blob):
        mask, at = blob[at:at + mask_width], at + mask_width
        row = []

        for column, datatype in enumerate(datatypes):
            is_null = mask[column // 8] >> (7 - column % 8) & 1

            if datatype == "char":
                length, = struct.unpack_from(">i", blob, at)
                at += 4
                value = blob[at:at + length].decode("utf-8")
                at += length
            elif datatype in NUMERIC:
                code, width = NUMERIC[datatype]
                value, = struct.unpack_from(code, blob, at)
                at += width
            else:
                raise ValueError(f"unsupported VOTable datatype: {datatype}")

            row.append(None if is_null else value)

        rows.append(row)

    return rows


def _decode_tabledata(table_data, datatypes: list[str]) -> list[list]:
    rows = []
    for tr in (e for e in table_data if _tag(e) == "TR"):
        cells = [e.text for e in tr if _tag(e) == "TD"]
        row = []
        for text, datatype in zip(cells, datatypes):
            if text is None or text == "":
                row.append(None)
            elif datatype == "char":
                row.append(text)
            elif datatype in ("short", "int", "long", "unsignedByte"):
                row.append(int(text))
            else:
                row.append(float(text))
        rows.append(row)
    return rows


def convert(vot_file: Path, csv_file: Path) -> int:
    """read the VOTable, reorder its columns to what data.py expects and write the csv"""
    root = ET.fromstring(_read_bytes(vot_file))

    status = next((e for e in root.iter() if _tag(e) == "INFO" and e.get("name") == "QUERY_STATUS"), None)
    if status is not None and status.get("value") == "ERROR":
        raise ValueError(f"the archive returned an error instead of a result:\n{(status.text or '').strip()}")

    names: list[str] = []
    datatypes: list[str] = []
    for field in (e for e in root.iter() if _tag(e) == "FIELD"):
        name, datatype = field.get("name"), field.get("datatype")
        if name is None or datatype is None:
            raise ValueError(f"a FIELD in the VOTable declares no name or datatype: {field.attrib}")
        names.append(name)
        datatypes.append(datatype)

    missing = [c for c in EXPECTED_COLUMNS if c not in names]
    if missing:
        raise ValueError(f"the VOTable is missing columns\n  missing: {missing}\n  got: {names}")

    stream = _first(root, "STREAM")
    table_data = _first(root, "TABLEDATA")

    if stream is not None:
        if stream.get("encoding") != "base64":
            raise ValueError(f"unsupported STREAM encoding: {stream.get('encoding')}")
        if _first(root, "BINARY2") is None:
            raise ValueError("only BINARY2 streams are supported - re-download as VOTable or CSV")
        rows = _decode_binary2(stream.text or "", datatypes)
    elif table_data is not None:
        rows = _decode_tabledata(table_data, datatypes)
    else:
        raise ValueError("found no BINARY2 stream and no TABLEDATA in the VOTable")

    order = [names.index(c) for c in EXPECTED_COLUMNS]
    types = [datatypes[i] for i in order]

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")  # csv defaults to crlf - the archive writes lf
        writer.writerow(EXPECTED_COLUMNS)
        writer.writerows([_format(row[i], t) for i, t in zip(order, types)] for row in rows)

    return len(rows)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: uv run gaia/votable_to_csv.py <downloaded.vot> [{CONFIG.csv_file}]")
        return 0

    vot_file = Path(argv[0]).expanduser()
    csv_file = Path(argv[1]).expanduser() if len(argv) > 1 else Path(CONFIG.csv_file)

    written = convert(vot_file, csv_file)
    print(f"{written} rows -> {csv_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
