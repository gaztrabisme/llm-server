#!/usr/bin/env python3
"""Graft MTP head tensors onto a base GGUF model.

Creates a new GGUF with the base model's weights + MTP prediction layer,
updating block_count and adding nextn_predict_layers metadata.

Usage:
    python3 scripts/graft-mtp.py \
        --base models/Qwen3.6-27B-UD-IQ3_XXS.gguf \
        --mtp models/27B_MTP.gguf \
        --out models/Qwen3.6-27B-MTP-UD-IQ3_XXS.gguf
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGMLQuantizationType, GGUFValueType


def extract_field_value(field):
    """Extract a Python value from a GGUFReader field."""
    vtype = field.types[0]

    if vtype == GGUFValueType.ARRAY:
        elem_type = field.types[1]
        count_part = field.parts[4]
        count = int(count_part[0]) if hasattr(count_part, '__len__') else int(count_part)

        if elem_type == GGUFValueType.STRING:
            return [bytes(field.parts[5 + i * 2 + 1]).decode('utf-8') for i in range(count)]
        else:
            return [_scalar(field.parts[5 + i]) for i in range(count)]

    if vtype == GGUFValueType.STRING:
        return bytes(field.parts[-1]).decode('utf-8')
    if vtype == GGUFValueType.BOOL:
        return bool(_scalar(field.parts[-1]))
    return _scalar(field.parts[-1])


def _scalar(part):
    """Extract a scalar value from a memmap part."""
    v = part.tolist()
    return v[0] if isinstance(v, list) else v


TYPE_TO_WRITER = {
    GGUFValueType.UINT8:   'add_uint8',
    GGUFValueType.UINT16:  'add_uint16',
    GGUFValueType.UINT32:  'add_uint32',
    GGUFValueType.UINT64:  'add_uint64',
    GGUFValueType.INT8:    'add_int8',
    GGUFValueType.INT16:   'add_int16',
    GGUFValueType.INT32:   'add_int32',
    GGUFValueType.INT64:   'add_int64',
    GGUFValueType.FLOAT32: 'add_float32',
    GGUFValueType.FLOAT64: 'add_float64',
    GGUFValueType.BOOL:    'add_bool',
    GGUFValueType.STRING:  'add_string',
    GGUFValueType.ARRAY:   'add_array',
}


def copy_field_to_writer(writer, name, field):
    """Copy a metadata field from reader to writer."""
    vtype = field.types[0]
    method_name = TYPE_TO_WRITER.get(vtype)
    if method_name is None:
        print(f"  WARNING: Skipping unsupported field type {vtype} for {name}")
        return
    value = extract_field_value(field)
    getattr(writer, method_name)(name, value)


def sha256_first_n(data, n=1024):
    """SHA256 of first N bytes for quick integrity check."""
    return hashlib.sha256(bytes(data[:n])).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Graft MTP head onto base GGUF")
    parser.add_argument("--base", required=True, help="Base model GGUF path")
    parser.add_argument("--mtp", required=True, help="MTP head GGUF path")
    parser.add_argument("--out", required=True, help="Output GGUF path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    base_path = Path(args.base)
    mtp_path = Path(args.mtp)
    out_path = Path(args.out)

    if out_path.exists() and not args.force:
        print(f"Output already exists: {out_path}")
        print("Use --force to overwrite")
        sys.exit(1)

    t0 = time.time()

    # --- Read inputs ---
    print(f"Reading base: {base_path} ({base_path.stat().st_size / 1e9:.1f} GB)")
    base = GGUFReader(str(base_path))
    print(f"  {len(base.tensors)} tensors, {len(base.fields) - 3} KV pairs")

    print(f"Reading MTP head: {mtp_path} ({mtp_path.stat().st_size / 1e6:.0f} MB)")
    mtp = GGUFReader(str(mtp_path))
    print(f"  {len(mtp.tensors)} tensors")

    arch = extract_field_value(base.fields['general.architecture'])
    block_count_key = f'{arch}.block_count'
    if block_count_key not in base.fields:
        print(f"  ERROR: No {block_count_key} in base model metadata")
        sys.exit(1)
    base_block_count = extract_field_value(base.fields[block_count_key])
    print(f"  Architecture: {arch}, block_count: {base_block_count}")

    # Verify MTP tensors are for the right block
    expected_prefix = f"blk.{base_block_count}."
    for t in mtp.tensors:
        if not t.name.startswith(expected_prefix):
            print(f"  ERROR: MTP tensor {t.name} doesn't match expected prefix {expected_prefix}")
            sys.exit(1)
    print(f"  All {len(mtp.tensors)} MTP tensors match block {base_block_count}")

    # --- Build output GGUF ---
    arch = extract_field_value(base.fields['general.architecture'])
    print(f"\nBuilding output GGUF (arch={arch})...")

    writer = GGUFWriter(str(out_path), arch)

    # Copy all metadata from base (skip internal GGUF fields and architecture which is auto-added)
    skip_fields = {'general.architecture'}
    override_fields = {block_count_key: base_block_count + 1}

    print("  Copying metadata...")
    field_count = 0
    for name, field in base.fields.items():
        if name.startswith('GGUF.'):
            continue
        if name in skip_fields:
            continue

        if name in override_fields:
            vtype = field.types[0]
            method_name = TYPE_TO_WRITER[vtype]
            getattr(writer, method_name)(name, override_fields[name])
            print(f"    {name}: {extract_field_value(field)} -> {override_fields[name]}")
        else:
            copy_field_to_writer(writer, name, field)
        field_count += 1

    # Add nextn_predict_layers
    nextn_key = f'{arch}.nextn_predict_layers'
    writer.add_uint32(nextn_key, 1)
    field_count += 1
    print(f"    {nextn_key}: 1 (added)")
    print(f"  Total KV pairs: {field_count + 1} (inc. architecture)")

    # --- Register tensors ---
    total_tensors = len(base.tensors) + len(mtp.tensors)
    print(f"\n  Registering {total_tensors} tensors ({len(base.tensors)} base + {len(mtp.tensors)} MTP)...")

    for i, t in enumerate(base.tensors):
        writer.add_tensor(
            t.name,
            t.data,
            raw_dtype=GGMLQuantizationType(t.tensor_type),
        )
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(base.tensors)} base tensors...")

    for t in mtp.tensors:
        writer.add_tensor(
            t.name,
            t.data,
            raw_dtype=GGMLQuantizationType(t.tensor_type),
        )

    print(f"    All {total_tensors} tensors registered")

    # --- Write output ---
    print(f"\n  Writing {out_path}...")
    writer.write_header_to_file()
    print("    Header written")
    writer.write_kv_data_to_file()
    print("    KV data written")
    writer.write_tensors_to_file()
    print("    Tensor info + data written")
    writer.close()

    elapsed = time.time() - t0
    out_size = out_path.stat().st_size
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {out_path} ({out_size / 1e9:.2f} GB)")

    # --- Verify ---
    print("\nVerifying output...")
    result = GGUFReader(str(out_path))
    print(f"  Tensors: {len(result.tensors)} (expected {total_tensors})")
    print(f"  block_count: {extract_field_value(result.fields[block_count_key])}")
    print(f"  nextn_predict_layers: {extract_field_value(result.fields[nextn_key])}")

    # Check first base tensor integrity
    base_t0 = base.tensors[0]
    result_t0 = result.tensors[0]
    base_hash = sha256_first_n(base_t0.data)
    result_hash = sha256_first_n(result_t0.data)
    print(f"  {base_t0.name} hash: {base_hash} -> {result_hash} {'OK' if base_hash == result_hash else 'MISMATCH!'}")

    # Check first MTP tensor integrity
    mtp_t0 = mtp.tensors[0]
    result_mtp = [t for t in result.tensors if t.name == mtp_t0.name][0]
    mtp_hash = sha256_first_n(mtp_t0.data)
    result_mtp_hash = sha256_first_n(result_mtp.data)
    print(f"  {mtp_t0.name} hash: {mtp_hash} -> {result_mtp_hash} {'OK' if mtp_hash == result_mtp_hash else 'MISMATCH!'}")

    ok = (
        len(result.tensors) == total_tensors
        and extract_field_value(result.fields[block_count_key]) == base_block_count + 1
        and nextn_key in result.fields
        and base_hash == result_hash
        and mtp_hash == result_mtp_hash
    )
    print(f"\n{'SUCCESS' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
