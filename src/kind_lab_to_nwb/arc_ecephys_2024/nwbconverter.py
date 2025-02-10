"""Primary NWBConverter class for this dataset."""

from neuroconv import (
    NWBConverter,
)
from neuroconv.datainterfaces import OpenEphysRecordingInterface, VideoInterface


class ArcEcephys2024NWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        OpenEphysRecording=OpenEphysRecordingInterface,
        Video=VideoInterface,
    )
