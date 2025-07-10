import os
from collections.abc import Callable, Sequence
from typing import Optional, Union

from napari.types import LayerData
from napari_builtins.io._read import magic_imread

PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], list[LayerData]]

METADATA_SUFFIXES = (
    "_cutoffs.txt",
    "_metadata.txt",
)


def get_reader(path: "PathOrPaths") -> Optional["ReaderFunction"]:
    # If we recognize the format, we return the actual reader function
    # If the path contains or points to a folder with tiff images, return the tiff reader.
    # get files and directories in the path
    if not os.path.exists(path) or not os.path.isdir(path):
        return None
    files = os.listdir(path)
    if any(f.endswith((".tif", ".tiff")) for f in files):
        return synchroton_tiff_reader
    for f in files:
        subpath = os.path.join(path, f)
        if os.path.isdir(subpath):
            subfiles = os.listdir(subpath)
            if any(sf.endswith((".tif", ".tiff")) for sf in subfiles):
                return synchroton_tiff_reader
    return None


def synchroton_tiff_reader(path: "PathOrPaths") -> list["LayerData"]:
    images = magic_imread(path)
    metadata = {}
    for md_suffix in METADATA_SUFFIXES:
        metadata_path = path + md_suffix
        if os.path.isfile(metadata_path):
            with open(metadata_path) as f:
                cutoffs = f.readlines()
            for line in cutoffs:
                line = line.removeprefix("#").replace(" ", "")
                name_val = line.split("=")
                if len(name_val) == 2:
                    name, val = name_val
                    try:
                        metadata[name] = float(val)
                    except ValueError:
                        continue
            break

    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        scan_name = os.path.basename(parent_dir)
        if "delta" in path:
            scan_name += "_delta"
        elif "beta" in path:
            scan_name += "_beta"
    else:
        scan_name = os.path.basename(path)

    return [(images, {"name": scan_name, "metadata": metadata})]
