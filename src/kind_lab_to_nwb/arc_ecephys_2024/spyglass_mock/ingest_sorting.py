"""Ingest mock ephys data from an NWB file into a spyglass database."""

import os
import datajoint as dj
from pathlib import Path
import pandas as pd
from pynwb import NWBHDF5IO
import numpy as np

dj.conn(use_tls=False)
dj_local_conf_path = "/home/alessandra/CatalystNeuro/kind-lab-to-nwb/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

# spyglass.common has the most frequently used tables
import spyglass.common as sgc  # this import connects to the database

# spyglass.data_import has tools for inserting NWB files into the database
import spyglass.data_import as sgi
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

# Spike sorting specific imports
from spyglass.spikesorting.spikesorting_merge import (
    SpikeSortingOutput,
)
import spyglass.spikesorting.v1 as sgs
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.spikesorting.analysis.v1.group import UnitSelectionParams
from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation


def main():
    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/mock_sorting.nwb")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)

    if sgc.Session & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Session & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        units_table = nwbfile.units.to_dataframe()
    insert_sorting(nwb_copy_file_name, units_table)

    print(UnitAnnotation().Annotation())
    group_key = {
        "nwb_file_name": nwb_copy_file_name,
        "sorted_spikes_group_name": "all_units",
    }
    group_key = (SortedSpikesGroup & group_key).fetch1("KEY")
    spikes_spyglass = SortedSpikesGroup().fetch_spike_data(group_key)
    spikes_nwb = [unit.spike_times for _, unit in units_table.iterrows()]
    for nwb_unit, spyglass_unit in zip(spikes_nwb, spikes_spyglass):
        np.testing.assert_array_equal(nwb_unit, spyglass_unit)


def insert_sorting(nwb_copy_file_name: str, units_table: pd.DataFrame):
    merge_id = str((SpikeSortingOutput.ImportedSpikeSorting & {"nwb_file_name": nwb_copy_file_name}).fetch1("merge_id"))

    UnitSelectionParams().insert_default()
    group_name = "all_units"
    SortedSpikesGroup().create_group(
        group_name=group_name,
        nwb_file_name=nwb_copy_file_name,
        keys=[{"spikesorting_merge_id": merge_id}],
    )
    annotation_to_type = {
        "custom_label": "label",
        "custom_quantification": "quantification",
    }
    group_key = {
        "nwb_file_name": nwb_copy_file_name,
        "sorted_spikes_group_name": group_name,
    }
    group_key = (SortedSpikesGroup & group_key).fetch1("KEY")
    _, unit_ids = SortedSpikesGroup().fetch_spike_data(group_key, return_unit_ids=True)

    for unit_key in unit_ids:
        unit_id = unit_key["unit_id"]
        for annotation, annotation_type in annotation_to_type.items():
            annotation_key = {
                **unit_key,
                "annotation": annotation,
                annotation_type: units_table.loc[unit_id][annotation],
            }
            UnitAnnotation().add_annotation(annotation_key, skip_duplicates=True)


if __name__ == "__main__":
    main()
    print("Done!")
