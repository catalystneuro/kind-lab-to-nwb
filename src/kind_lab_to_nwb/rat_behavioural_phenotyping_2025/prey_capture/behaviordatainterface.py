from pathlib import Path
from typing import List, Optional, Union

from hdmf.common.table import DynamicTable
from pymatreader import read_mat
from pynwb import NWBFile

from neuroconv import BaseDataInterface
from neuroconv.tools import nwb_helpers


class PreyCaptureBehavioralInterface(BaseDataInterface):
    """Adds USV detection scores as analysis."""

    def __init__(self, file_paths: List[Union[str, Path]]):
        super().__init__(file_paths=file_paths)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: Optional[dict], **conversion_options) -> None:
        detections_table = DynamicTable(name="detections_table", description="The USV detections table.")
        detections_table.add_column(name="accept", description="Whether the USV was accepted based on manual review")
        detections_table.add_column(name="score", description="The score of the USV detection")
        detections_table.add_column(name="box", description="Bounding box coordinates")  # TODO: ask to confirm
        for file_path in self.source_data["file_paths"]:
            data = read_mat(file_path)

            if "Calls" not in data:
                continue

            detections_table.add_row(
                accept=bool(int(data["Calls"]["Accept"])),
                score=float(data["Calls"]["Score"]),
                box=data["Calls"]["Box"],
            )

        behavior = nwb_helpers.get_module(nwbfile, name="behavior", description="Contains the USV detection scores.")
        behavior.add(detections_table)
