# %%
from pathlib import Path

from neuroconv.tools.path_expansion import LocalPathExpander

data_dir_path = Path("D:/Kind-CN-data-share/neuronal_circuits/fear_conditionning_paradigm")
output_dir_path = Path("D:/kind_lab_conversion_nwb")

source_data_spec = {
    "OpenEphysRecording": {
        "base_directory": data_dir_path,
        "folder_path": "{subject_id}/{session_id}/{ff}",
    },
    "Video": {"base_directory": data_dir_path, "file_path": "{subject_id}/{session_id}/{other}.avi"},
}
for interface, source_data in source_data_spec.items():
    print(interface, source_data)
# %%
# Instantiate LocalPathExpander
path_expander = LocalPathExpander()

# Expand paths and extract metadata
metadata_list = path_expander.expand_paths(source_data_spec)

# %%
for i, sub in enumerate(metadata_list):
    if sub["metadata"]["Subject"]["subject_id"] == "Rat_700":
        print(i, sub["metadata"]["Subject"]["subject_id"])
# %%
from neuroconv.datainterfaces import OpenEphysRecordingInterface

OpenEphysRecordingInterface.get_stream_names(
    folder_path=metadata_list[0]["source_data"]["OpenEphysRecording"]["folder_path"]
)

# %%
