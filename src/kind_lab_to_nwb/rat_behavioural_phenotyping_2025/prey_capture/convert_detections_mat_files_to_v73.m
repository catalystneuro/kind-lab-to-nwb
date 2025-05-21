function convert_detections_mat_files_to_v73(folder_path)
    % Find all .mat files in the directory and subdirectories
    matFiles = dir(fullfile(folder_path, '**/*.mat'));

    % Loop through each file
    for i = 1:length(matFiles)
        % Get full path to the file
        filePath = fullfile(matFiles(i).folder, matFiles(i).name);

        matFilesFolderPath = fullfile(matFiles(i).folder, 'converted');
        % Create a 'converted' directory in the matFiles folder
        if ~exist(matFilesFolderPath, 'dir')
            mkdir(matFilesFolderPath);
        end

        try
            % Load the MAT file
            data = load(filePath);

            % Check if 'Calls' exists and is a table
            if isfield(data, 'Calls') && istable(data.Calls)
                % Create the output filename in the new location
                outputPath = fullfile(matFilesFolderPath, matFiles(i).name);


                % Extract Calls
                Calls = data.Calls;
                Calls = table2struct(Calls);

                % Save with -v7.3 flag
                if isfield(data, 'audiodata')
                    audiodata = data.audiodata;
                    save(outputPath, 'audiodata', 'Calls', '-v7.3');
                else
                    save(outputPath, 'Calls', '-v7.3');
                end
                clear data

                fprintf('Successfully processed: %s\n', outputPath);
            else
                fprintf('Skipping (no Calls table): %s\n', filePath);
            end

        catch ME
            % If there's an error, print it and continue with next file
            fprintf('Error processing %s: %s\n', filePath, ME.message);
            continue;
        end
    end

    fprintf('Processing complete!\n');
end
