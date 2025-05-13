import subprocess
from pathlib import Path
from typing import Union


def run_matlab_convert_detections_mat_files_to_v73(
    matlab_path: Union[str, Path],
    folder_path: Union[str, Path],
) -> None:
    """
    This function executes a MATLAB script to process `.mat` files within a specified folder.
    It converts `.mat` files containing `Calls` as a table into a structure and saves the converted
    files in the designated folder in MATLAB v7.3 format. The conversion process checks for the
    existence of a `Calls` table and, if present, also saves additional `audiodata` if available.
    Errors during processing or the absence of necessary fields in `.mat` files are logged.

    Parameters
    ----------
    matlab_path : Union[str, Path]
        Path to the MATLAB executable.
    folder_path : Union[str, Path]
        Path to the root folder containing the `.mat` files to process. The function
        processes files recursively within this folder. Converted files are saved in a subdirectory
        called `converted` within the respective `.mat` file's folder.
    """
    if not Path(matlab_path).exists():
        raise FileNotFoundError(f"MATLAB executable not found at {matlab_path}. Please verify the installation path.")

    # Create the MATLAB command as a single string with proper formatting
    matlab_script = (
        f"folder_path='{folder_path}';matFiles=dir(fullfile(folder_path,'**/*.mat'));"
        "for i=1:length(matFiles);filePath=fullfile(matFiles(i).folder,matFiles(i).name);"
        "matFilesFolderPath=fullfile(matFiles(i).folder,'converted');"
        "if ~exist(matFilesFolderPath,'dir');mkdir(matFilesFolderPath);end;"
        "try;data=load(filePath);"
        "if isfield(data,'Calls')&&istable(data.Calls);"
        "outputPath=fullfile(matFilesFolderPath,matFiles(i).name);"
        "Calls=data.Calls;Calls=table2struct(Calls);"
        "if isfield(data,'audiodata');audiodata=data.audiodata;"
        "save(outputPath,'audiodata','Calls','-v7.3');"
        "else;save(outputPath,'Calls','-v7.3');end;"
        "clear data;fprintf('Successfully processed: %s\\n',outputPath);"
        "else;fprintf('Skipping (no Calls table): %s\\n',filePath);end;"
        "catch ME;fprintf('Error processing %s: %s\\n',filePath,ME.message);"
        "continue;end;end;fprintf('Processing complete!\\n');exit;"
    )

    # Combine the script into a single command string
    command_str = f"try;{matlab_script}catch ME;disp(ME.message);end;exit;"

    matlab_cmd = [
        str(matlab_path),
        "-nodisplay",
        "-nosplash",
        "-nodesktop",
        "-batch",  # Using -batch instead of -r for better command handling
        command_str,
    ]

    subprocess.run(
        matlab_cmd,
        check=True,
    )


if __name__ == "__main__":
    # Update this path according to your MATLAB installation
    matlab_path = "/Applications/MATLAB_R2024b.app/bin/matlab"
    folder_path = "/Volumes/T9/Behavioural Pipeline/Prey Capture/Arid1b/Arid1b(1)_PC/TestD4"
    run_matlab_convert_detections_mat_files_to_v73(matlab_path=matlab_path, folder_path=folder_path)
