"""Create a mock NWB file with spyglass-compatible LFP data for testing purposes."""

from typing import Optional
import numpy as np

from pynwb import NWBFile
from pynwb.ecephys import LFP

from ndx_franklab_novela import DataAcqDevice, Probe, Shank, ShanksElectrode, NwbElectrodeGroup


def add_lfp(nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None) -> None:

    data_acq_device = DataAcqDevice(**metadata["Devices"]["DataAcqDevice"])
    nwbfile.add_device(data_acq_device)

    electrode = ShanksElectrode(**metadata["Ecephys"]["ShanksElectrode"])
    shanks_electrodes = [electrode]
    shank = Shank(**metadata["Ecephys"]["Shank"], shanks_electrodes=shanks_electrodes)
    probe = Probe(**metadata["Ecephys"]["Probe"], shanks=[shank])
    nwbfile.add_device(probe)

    # add to electrical series
    electrode_group = NwbElectrodeGroup(**metadata["Ecephys"]["NwbElectrodeGroup"], device=probe)
    nwbfile.add_electrode_group(electrode_group)

    extra_cols = [
        "probe_shank",
        "probe_electrode",
        "bad_channel",
        "ref_elect_id",
    ]
    for col in extra_cols:
        nwbfile.add_electrode_column(name=col, description=f"description for {col}")

    # TODO: from "D:\Kind-CN-data-share\neuronal_circuits\fear_conditionning_paradigm\channels_details_v2.xlsx" get electrode location and bad channel info
    # each row in the excel file corresponds to a subject
    # each row contains the following columns: ID, Folder, Genotype, source_number , 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    # the column folder is the subject_id
    # the last 16 columns are the channel location. If the value is "bad", the channel is  bad_channel=True
    # for n in range(number_of_channels):
    # if df[df["Folder"] == subject_id][str(n)].iloc[0] == "bad":
    #     bad_channel = True
    #     location = None
    # else:
    #     bad_channel = False
    #     location = df[df["Folder"] == subject_id][str(n)].iloc[0]
    # nwbfile.add_electrode(
    #     location=location,
    #     group=electrode_group,
    #     probe_shank=1,
    #     probe_electrode=1,
    #     bad_channel=bad_channel,
    #     ref_elect_id=n,
    #     x=0.0, # WHERE TO GET THIS INFO?
    #     y=0.0, # WHERE TO GET THIS INFO?
    #     z=0.0, # WHERE TO GET THIS INFO?
    # electrodes = nwbfile.electrodes.create_region(
    #     name="electrodes", region=list(range(number_of_channels)), description="electrodes"
    # )

    lfp_electrodes = nwbfile.electrodes.create_region(
        name="electrodes", region=list(range(number_of_channels)), description="lfp electrodes"
    )
    # TODO extract lfp_eseries from OpenEphysRecordingInterface
    lfp = LFP(electrical_series=lfp_eseries)
    ecephys_module = nwbfile.create_processing_module(name="ecephys", description="ecephys module")
    ecephys_module.add(lfp)
