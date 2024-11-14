"""Primary NWBConverter class for this dataset."""

from neuroconv import (
    NWBConverter,
)
from neuroconv.datainterfaces import (
    PhySortingInterface,
    SpikeGLXRecordingInterface,
)

from .arc_ecephys_2024 import (
    ArcEcephys2024BehaviorInterface,
)


class ArcEcephys2024NWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=SpikeGLXRecordingInterface,
        Sorting=PhySortingInterface,
        Behavior=ArcEcephys2024BehaviorInterface,
    )
