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
conda activate kind-lab-to-nwb-env
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

**For detailed information about the experimental procedure, subjects, data streams, devices, temporal alignment, and Spyglass compatibility, see [arc_ecephys_2024/notes.md](src/kind_lab_to_nwb/arc_ecephys_2024/notes.md).**

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

The module includes specialized utilities for Spyglass compatibility

The `spyglass_mock/` directory contains utilities for testing Spyglass integration and mock data generation.

### rat_behavioural_phenotyping_2025

This module contains conversion scripts for various behavioral phenotyping experiments as part of the Rat Behavioural Phenotyping Pipeline developed at SIDB (Simons Initiative for the Developing Brain). The pipeline includes:

- **Auditory Fear Conditioning**: Fear learning paradigms
- **Marble Interaction**: Object interaction behaviors  
- **Object Location Memory**: Spatial memory tasks
- **Object Recognition**: Recognition memory tests
- **One Trial Social**: Social behavior assessment
- **Prey Capture**: Predatory behavior analysis
- **Water Maze**: Spatial navigation tasks

**For detailed information about experimental procedures, apparatus, analysis methods, and data streams for each behavioral task, see [rat_behavioural_phenotyping_2025/notes.md](src/kind_lab_to_nwb/rat_behavioural_phenotyping_2025/notes.md).**

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


# Spyglass 
## Spyglass Setup Instructions

### Before Installing Mamba
I recommend uninstalling any Conda and/or Mamba distributions that you have installed already. This will help avoid issues with competing conda/mamba locations. 

### Mamba* (on MacOS)
1. Install Homebrew
2. Install mambaforge: `brew install --cask mambaforge`
3. Run `mamba info` to check that mamba is working*
    a. If you get `zsh: command not found: mamba`, you might need to update your .zshrc or .bashrc file
*There are several different ways to install Mamba, including Miniforge and conda-forge. To my knowledge, all of them work equally well. 

### Spyglass Environment
1. Navigate to the Spyglass directory
2. Create the Spyglass environment as specified: `mamba env create -f environment.yml`
3. Activate the Spyglass environment: `mamba activate spyglass`
4. Install kachery-cloud: `pip install kachery-cloud`

### Spyglass Setup

1. Run the docker container: `docker run --name spyglass-db -p 3306:3306 -e MYSQL_ROOT_PASSWORD=tutorial datajoint/mysql:8.0`
2. Open [00_Setup.ipynb](https://github.com/LorenFrankLab/spyglass/blob/master/notebooks/00_Setup.ipynb) in VSCode
3. In the first code cell,
    - change base_dir to a new directory where you would like to store all the spyglass output files. This should be different from the directory where the spyglass git repo is stored.
    - change database_user to “root”
    - change database_password to “tutorial”
    - change database_host to “localhost”
4. Run the first code cell
    - When prompted,
        a. enter user name: root
        b. enter password: tutorial
        c. Update local setting: yes
        d. Replace existing file: yes
    - This will create a dj_local_conf.json file with all the relevant info for spyglass configuration
5. In the dj_local_conf.json file (Line 15), set database.use_tls to false 
6. On Docker Desktop, stop and delete the spyglass-db container
7. Run a new docker container with a mounted volume: docker run --name spyglass-db -v dj-vol:/var/lib/mysql -p 3306:3306 -e MYSQL_ROOT_PASSWORD=tutorial datajoint/mysql
8. If you HAVE already converted the .nwb files --> move/copy them to base_dir/raw (where base_dir is the directory that you chose for Step 3).
