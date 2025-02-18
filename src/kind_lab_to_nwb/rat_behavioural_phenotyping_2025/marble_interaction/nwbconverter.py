"""Primary NWBConverter class for this dataset."""

from neuroconv import (
    NWBConverter,
)
from neuroconv.datainterfaces import (
    VideoInterface,
)

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import (
    BORISBehavioralEventsInterface,
)


class MarbleInteractionNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        MarbleInteractionBehavior=BORISBehavioralEventsInterface,
        Video=VideoInterface,
    )
