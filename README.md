# kind-lab-to-nwb

NWB conversion scripts for Kind lab data to the [Neurodata Without Borders](https://nwb-overview.readthedocs.io/) data format.

This repository contains conversion pipelines for multiple datasets from the Kind lab, each organized in its own module with specific conversion scripts and utilities.

## Repository Structure

```
kind-lab-to-nwb/
├── LICENSE
├── make_env.yml
├── pyproject.toml
├── README.md
├── MANIFEST.in
├── .gitignore
├── .pre-commit-config.yaml
├── dj_local_conf.json
├── misc/
└── src/
    └── kind_lab_to_nwb/
        ├── __init__.py
        ├── arc_ecephys_2024/           # Fear conditioning EEG/LFP dataset
        │   ├── __init__.py
        │   ├── convert_session.py      # Single session conversion
        │   ├── convert_all_sessions.py # Batch conversion script
        │   ├── insert_session.py       # Database insertion utilities
        │   ├── metadata.yaml           # Experiment metadata
        │   ├── notes.md                # Detailed conversion notes
        │   ├── spyglass_mock/          # Spyglass compatibility utilities
        │   ├── tutorial/               # Data access tutorials
        │   └── utils/                  # Conversion utility functions
        └── rat_behavioural_phenotyping_2025/  # Behavioral phenotyping datasets
            ├── auditory_fear_conditioning/
            ├── marble_interaction/
            ├── object_location_memory/
            ├── object_recognition/
            ├── one_trial_social/
            ├── prey_capture/
            ├── water_maze/
            ├── interfaces/             # Custom data interfaces
            ├── tutorials/              # Analysis tutorials
            └── utils/                  # Shared utilities
```

## Installation

### Installation from GitHub

You can install the package with:

```bash
git clone https://github.com/catalystneuro/kind-lab-to-nwb
cd kind-lab-to-nwb
conda env create --file make_env.yml
conda activate kind_lab_to_nwb_env
```

Alternatively, using pip only:

```bash
git clone https://github.com/catalystneuro/kind-lab-to-nwb
cd kind-lab-to-nwb
pip install --editable .
```

## Datasets

### arc_ecephys_2024: Fear Conditioning EEG/LFP Dataset

This module converts data from a fear conditioning experiment investigating neural responses in Syngap1+/Delta-GAP rats. The dataset includes EEG, LFP, accelerometer, and behavioral video recordings across multiple experimental sessions.

#### Dataset Description

**Experiment**: Fear conditioning paradigm in male wild-type and Syngap1+/Delta-GAP rats (n=31, ages 3-6 months)

**Data Types**:
- Local Field Potentials (LFP) - 2 kHz sampling, 0.1-600 Hz bandpass
- Electroencephalogram (EEG) - surface recordings from multiple brain regions
- 3-axis accelerometer data - 500 Hz sampling
- Behavioral videos - ~30 Hz
- TTL trigger events - 2 kHz sampling

**Experimental Sessions**:
1. **Hab_1**: Context habituation (Day 1)
2. **Seizure_screening**: Seizure monitoring (Day 2, subset of animals)
3. **Hab_2**: Second context habituation (Day 3)
4. **Baseline_tone_flash_hab**: CS pre-exposure (Day 4, subset)
5. **Cond**: Fear conditioning with CS-US pairings (Day 4)
6. **Recall**: Fear response testing (Day 5)

#### Quick Start

##### Converting a Single Session

```python
from pathlib import Path
from kind_lab_to_nwb.arc_ecephys_2024.convert_session import session_to_nwb
from neuroconv.tools.path_expansion import LocalPathExpander

# Set up paths
data_dir_path = Path("/path/to/your/data")
output_dir_path = Path("/path/to/output")

# Define source data specification
source_data_spec = {
    "OpenEphysRecording": {
        "base_directory": data_dir_path,
        "folder_path": "{subject_id}/{session_id}/{subject_id}_{session_date}_{session_time}_{task}",
    },
}

# Expand paths and extract metadata
path_expander = LocalPathExpander()
metadata_list = path_expander.expand_paths(source_data_spec)

# Convert first session
session_to_nwb(
    data_dir_path=data_dir_path,
    output_dir_path=output_dir_path,
    path_expander_metadata=metadata_list[0],
    stub_test=False,
    verbose=True,
)
```

##### Converting All Sessions

```python
from kind_lab_to_nwb.arc_ecephys_2024.convert_all_sessions import dataset_to_nwb

dataset_to_nwb(
    data_dir_path="/path/to/your/data",
    output_dir_path="/path/to/output",
    verbose=True,
)
```

#### Data Access Tutorial

The module includes a comprehensive Jupyter notebook tutorial (`tutorial/access_data_tutorial.ipynb`) demonstrating how to:

- Stream NWB files from DANDI Archive
- Access EEG and LFP signals
- View behavioral videos
- Analyze accelerometer data
- Extract TTL trigger events
- Visualize multi-modal data

#### Expected Data Structure

Your raw data should be organized as follows:

```
data_directory/
├── channels_details_v2.xlsx          # Electrode configuration
├── cs_video_frames.xlsx              # Video synchronization data
├── subject_id_1/
│   └── session_id/
│       ├── subject_id_session_date_session_time_task/  # OpenEphys data
│       └── video_file.avi            # Behavioral video
└── subject_id_2/
    └── session_id/
        └── ...
```

#### Key Features

- **Time Alignment**: Automatic synchronization between electrophysiology, video, and behavioral triggers
- **Multi-modal Integration**: Combines EEG, LFP, accelerometer, and video data in single NWB files
- **Spyglass Compatibility**: Includes utilities for integration with the Spyglass analysis framework
- **Flexible Processing**: Supports both single session and batch conversion workflows
- **Rich Metadata**: Comprehensive experimental metadata including electrode locations and task descriptions

#### Spyglass Integration

The module includes specialized utilities for Spyglass compatibility:

```python
# For Spyglass integration, use the custom branch
git clone -b populate_sensor_data https://github.com/alessandratrapani/spyglass.git
```

The `spyglass_mock/` directory contains utilities for testing Spyglass integration and mock data generation.

### rat_behavioural_phenotyping_2025

This module contains conversion scripts for various behavioral phenotyping experiments including:

- **Auditory Fear Conditioning**: Fear learning paradigms
- **Marble Interaction**: Object interaction behaviors  
- **Object Location Memory**: Spatial memory tasks
- **Object Recognition**: Recognition memory tests
- **One Trial Social**: Social behavior assessment
- **Prey Capture**: Predatory behavior analysis
- **Water Maze**: Spatial navigation tasks

Each sub-module includes its own conversion scripts, metadata files, and analysis tutorials.

## Usage Guidelines

### General Workflow

1. **Prepare your data** according to the expected structure for your dataset
2. **Install the package** and dependencies
3. **Configure paths** in the conversion scripts
4. **Run conversion** (single session or batch)
5. **Validate output** using the provided tutorials

### Customization

Each conversion module can be customized by:

- Modifying `metadata.yaml` files for experiment-specific metadata
- Adjusting conversion parameters in the main scripts
- Adding custom interfaces for new data types
- Extending utility functions for specific analysis needs

## Contributing

When adding new conversion modules:

1. Create a new directory under `src/kind_lab_to_nwb/`
2. Include the standard files: `convert_session.py`, `metadata.yaml`, `notes.md`
3. Add utility functions in a `utils/` subdirectory
4. Provide tutorials and documentation
5. Follow the existing code structure and naming conventions

## Support

For questions about specific conversions or to report issues, please:

1. Check the `notes.md` file in the relevant module
2. Review the tutorial notebooks
3. Open an issue on the GitHub repository

## References

- **arc_ecephys_2024**: Based on Katsanevaki, D., et al. (2024). "Key roles of C2/GAP domains in SYNGAP1-related pathophysiology." Cell Reports, 43(9), 114733.
- **DANDI Archive**: Dataset available at [DANDI:001457](https://dandiarchive.org/dandiset/001457/draft)
- **EMBER Archive**: Datasets available at:
  - [EMBER:000199](https://dandi.emberarchive.org/dandiset/000199) - Auditory Fear Conditioning
  - [EMBER:000200](https://dandi.emberarchive.org/dandiset/000200) - Marble Interaction
  - [EMBER:000201](https://dandi.emberarchive.org/dandiset/000201) - Object Location Memory
  - [EMBER:000202](https://dandi.emberarchive.org/dandiset/000202) - Object Recognition
  - [EMBER:000203](https://dandi.emberarchive.org/dandiset/000203) - One Trial Social
  - [EMBER:000204](https://dandi.emberarchive.org/dandiset/000204) - Prey Capture
  - [EMBER:000205](https://dandi.emberarchive.org/dandiset/000205) - Water Maze
