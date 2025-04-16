"""Primary NWBConverter class for this dataset."""

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.auditory_fear_conditioning.behaviordatainterface import (
    AuditoryFearConditioningBehavioralInterface,
)
from neuroconv import NWBConverter
from neuroconv.datainterfaces import VideoInterface


class AuditoryFearConditioningNWBConverter(NWBConverter):
    """NWBConverter for the auditory fear conditioning dataset."""

    data_interface_classes = dict(
        Video=VideoInterface,
        Behavior=AuditoryFearConditioningBehavioralInterface,
    )
