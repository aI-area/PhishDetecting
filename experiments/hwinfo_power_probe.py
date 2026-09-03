#!/usr/bin/env python3
"""Read HWiNFO SM2 power readings without changing HWiNFO settings."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import sys


MAP_NAME = r"Global\HWiNFO_SENS_SM2"
FILE_MAP_READ = 0x0004
MAGIC = int.from_bytes(b"HWiS", "little")


class Header(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("interface_version", ctypes.c_uint32),
        ("last_update", ctypes.c_int64),
        ("sensor_offset", ctypes.c_uint32),
        ("sensor_size", ctypes.c_uint32),
        ("sensor_count", ctypes.c_uint32),
        ("reading_offset", ctypes.c_uint32),
        ("reading_size", ctypes.c_uint32),
        ("reading_count", ctypes.c_uint32),
    ]


class Reading(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("reading_type", ctypes.c_uint32),
        ("sensor_index", ctypes.c_uint32),
        ("reading_id", ctypes.c_uint32),
        ("name_original", ctypes.c_char * 128),
        ("name_user", ctypes.c_char * 128),
        ("unit", ctypes.c_char * 16),
        ("value", ctypes.c_double),
        ("value_min", ctypes.c_double),
        ("value_max", ctypes.c_double),
        ("value_avg", ctypes.c_double),
    ]


def decode(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


def fail(message: str, winerror: int | None = None) -> None:
    print(json.dumps({"status": "FAIL", "message": message, "winerror": winerror}, indent=2))
    raise SystemExit(1)


def main() -> None:
    if sys.platform != "win32":
        fail("This probe requires Windows")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_mapping = kernel32.OpenFileMappingW
    open_mapping.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
    open_mapping.restype = wt.HANDLE
    map_view = kernel32.MapViewOfFile
    map_view.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t]
    map_view.restype = wt.LPVOID
    unmap_view = kernel32.UnmapViewOfFile
    unmap_view.argtypes = [wt.LPCVOID]
    unmap_view.restype = wt.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wt.HANDLE]
    close_handle.restype = wt.BOOL

    handle = open_mapping(FILE_MAP_READ, False, MAP_NAME)
    if not handle:
        fail("HWiNFO shared memory is unavailable", ctypes.get_last_error())
    view = map_view(handle, FILE_MAP_READ, 0, 0, 0)
    if not view:
        error = ctypes.get_last_error()
        close_handle(handle)
        fail("HWiNFO shared memory could not be mapped", error)

    try:
        address = int(ctypes.cast(view, ctypes.c_void_p).value)
        header = Header.from_buffer_copy(ctypes.string_at(address, ctypes.sizeof(Header)))
        if header.magic != MAGIC:
            fail(f"Unexpected HWiNFO signature 0x{header.magic:08x}")
        if header.reading_size < ctypes.sizeof(Reading):
            fail(
                f"Unsupported HWiNFO reading record size {header.reading_size}; "
                f"expected at least {ctypes.sizeof(Reading)}"
            )
        readings = []
        base = address + header.reading_offset
        for index in range(header.reading_count):
            record_address = base + index * header.reading_size
            record = Reading.from_buffer_copy(
                ctypes.string_at(record_address, ctypes.sizeof(Reading))
            )
            original = decode(bytes(record.name_original))
            user = decode(bytes(record.name_user))
            unit = decode(bytes(record.unit))
            if unit == "W" or "power" in original.lower() or "power" in user.lower():
                readings.append(
                    {
                        "index": index,
                        "sensor_index": int(record.sensor_index),
                        "name_original": original,
                        "name_user": user,
                        "unit": unit,
                        "value": float(record.value),
                    }
                )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mapping": MAP_NAME,
                    "header_version": int(header.version),
                    "header_interface_version": int(header.interface_version),
                    "reading_size": int(header.reading_size),
                    "reading_count": int(header.reading_count),
                    "power_readings": readings,
                },
                indent=2,
            )
        )
    finally:
        unmap_view(view)
        close_handle(handle)


if __name__ == "__main__":
    main()
