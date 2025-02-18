"""Primary NWBConverter class for Object Location Memory experiment."""

from neuroconv import NWBConverter
from neuroconv.datainterfaces import VideoInterface
from .behaviorinterface import BORISBehavioralInterface


class ObjectLocationMemoryNWBConverter(NWBConverter):
    """Primary conversion class for Object Location Memory behavioral dataset."""

    data_interface_classes = dict(
        Video=VideoInterface,
        BehavioralScoring=BORISBehavioralInterface,
    )

    def __init__(self, source_data):
        """
        Initialize the ObjectLocationMemoryNWBConverter.

        Parameters
        ----------
        source_data : dict
            Dictionary with paths to source files. Should contain keys:
            - 'Video': dict with 'file_path' pointing to video file (.ts or .mkv)
            - 'BehavioralScoring': dict with 'file_path' pointing to BORIS scoring file (.boris or .xls)
        """
        super().__init__(source_data)
