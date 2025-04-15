"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter
from neuroconv.datainterfaces import VideoInterface


class AuditoryFearConditioningNWBConverter(NWBConverter):
    """NWBConverter for the auditory fear conditioning dataset."""

    data_interface_classes = dict(
        Video=VideoInterface,
    )
