"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter
from neuroconv.datainterfaces import ExternalVideoInterface

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import (
    BORISBehavioralEventsInterface,
)


class ObjectLocationMemoryNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Video=ExternalVideoInterface,
        SampleVideo=ExternalVideoInterface,
        TestVideo=ExternalVideoInterface,
        TestObjectRecognitionBehavior=BORISBehavioralEventsInterface,
        SampleObjectRecognitionBehavior=BORISBehavioralEventsInterface,
    )

    # TODO align sample and test video to the session start time
