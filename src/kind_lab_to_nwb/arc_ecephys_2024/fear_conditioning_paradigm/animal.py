import os

import pandas as pd


class Animal:
    def __init__(self, animal_id, params):
        self.id = animal_id
        self.params = params
        # Additional initialization as needed

        self.animal_processing()

    def _update_indices(self, dictionary):
        # Remove the bad channels
        for channel in sorted(self.bad_channels, reverse=True):
            del dictionary[channel]

        # Update the indices of the remaining channels
        updated_dict = {}
        for i, key in enumerate(sorted(dictionary.keys())):
            updated_dict[i] = dictionary[key]

        return updated_dict

    def areas_dict_cleanup(self):
        # remove the bad channels entries from tha animals areas to channel index dictionary
        areas_animal_full = {}
        self.bad_channels = []
        for ch_idx in self.animal_info.iloc[:, 4:].columns:
            areas_animal_full[ch_idx] = self.animal_info[ch_idx].values[0]
            if self.animal_info[ch_idx].values[0] == 'bad':
                self.bad_channels.append(ch_idx)
        self.areas_animal_clean = self._update_indices(areas_animal_full)

    def extract_animal_recorded_sessions(self):
        self.animal_recorded_sessions = []

        # List all entries in the base folder
        for entry in os.listdir(self.animal_base_folder):
            subfolder_path = os.path.join(self.animal_base_folder, entry)

            # Check if the entry is a directory
            if os.path.isdir(subfolder_path):
                # Walk through the directory tree
                for root, dirs, files in os.walk(subfolder_path):
                    # Check if any .continuous files exist in this path
                    if any(file.endswith('.continuous') for file in files):
                        self.animal_recorded_sessions.append(entry)
                        break  # No need to continue once a file is found


    def animal_processing(self):
        assert isinstance(self.params.all_animals_info,
                          pd.DataFrame), "the xlsx electrode file must be loaded as a pandas DataFrame"

        self.animal_info = self.params.all_animals_info.loc[self.params.all_animals_info['ID'] == self.id]
        self.animal_name = self.animal_info['Folder'].values[0]
        self.genotype = self.animal_info['Genotype'].values[0]

        assert self.genotype in ['wt',
                                 'het'], "Pipeline not build for more than 2 different genotypes names wt and het yet"

        self.src = str(self.animal_info['source_number'].values[0])
        print('Processing {}'.format(self.animal_name))
        self.animal_base_folder = self.params.base_dir + str(self.animal_name) + '/'
        self.extract_animal_recorded_sessions()
        assert os.path.isdir(self.animal_base_folder), f"Can't find animal data folder : {self.animal_base_folder}"

        self.areas_dict_cleanup()

        assert len(self.areas_animal_clean), "all channels are labeled 'bad' in the xlsx electrode file"
