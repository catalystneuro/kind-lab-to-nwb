import os
from pathlib import Path
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.NWB_conversion_utils import extract_session_datetime_from_xml
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.animal import Animal
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.params import AnalysisParams
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.session import Session


class Pipeline:
    def __init__(self, params):
        self.params = params

    def run(self):
        skip = 0
        for animal_id in self.params.all_animals_info['ID'][skip:]:
            animal = Animal(animal_id, self.params)
            for session_folder in self.params.sessions_folders:
                if session_folder in animal.animal_recorded_sessions:
                    session = Session(self.params, animal, session_folder)
                    session.compute_oscillations_analysis()
                    # session.raw.plot(block=True)

    def extract_kwargs_NWB_conversion(self, data_dir_path, output_dir_path):

        session_to_nwb_kwargs_per_session = []

        for animal_id in self.params.all_animals_info['ID'][:]:
            animal = Animal(animal_id, self.params)
            for session_folder in self.params.sessions_folders:
                if session_folder in animal.animal_recorded_sessions:
                    session = Session(self.params, animal, session_folder)

                    session_to_nwb_kwargs = {}

                    session_to_nwb_kwargs["data_dir_path"] = data_dir_path
                    session_to_nwb_kwargs["output_dir_path"] = output_dir_path
                    session_to_nwb_kwargs["video_starting_time"] = None

                    session_to_nwb_kwargs["path_expander_metadata"] = {}
                    session_to_nwb_kwargs["path_expander_metadata"]['metadata'] = {}
                    session_to_nwb_kwargs["path_expander_metadata"]['metadata']['Subject'] = {}
                    session_to_nwb_kwargs["path_expander_metadata"]['metadata']['Subject'][
                        'subject_id'] = animal.animal_name

                    session_to_nwb_kwargs["path_expander_metadata"]['metadata']['NWBFile'] = {}
                    session_to_nwb_kwargs['path_expander_metadata']['metadata']["NWBFile"][
                        'session_id'] = session_folder
                    session_to_nwb_kwargs['path_expander_metadata']['metadata']['NWBFile']['session_start_time'] = \
                        session.recording.info['meas_date']
                    session_to_nwb_kwargs["path_expander_metadata"]['metadata']['extras'] = {}

                    dt = extract_session_datetime_from_xml(session.session_dir)
                    date_str = dt.strftime("%Y-%m-%d")  # e.g., 2019-10-28
                    time_str = dt.strftime("%H-%M-%S")

                    session_to_nwb_kwargs["path_expander_metadata"]['metadata']['extras']['session_date'] = date_str
                    session_to_nwb_kwargs["path_expander_metadata"]['metadata']['extras']['session_time'] = time_str

                    session_to_nwb_kwargs["path_expander_metadata"]["source_data"] = {}
                    session_to_nwb_kwargs["path_expander_metadata"]["source_data"]["OpenEphysRecording"] = {}

                    data_session_folder = os.listdir(session.session_dir)
                    data_session_folder = [i for i in data_session_folder if
                                           i.startswith(str(session.animal.animal_name))]
                    data_session_folder = [i for i in data_session_folder if '.' not in i][0]
                    data_session_folder = session.session_dir + data_session_folder + '/'
                    rec_node_folder = [i for i in os.listdir(data_session_folder) if 'Record Node' in i]
                    data_session_folder = data_session_folder + rec_node_folder[0]
                    session_to_nwb_kwargs["path_expander_metadata"]["source_data"]["OpenEphysRecording"][
                        "folder_path"] = data_session_folder

                    session_to_nwb_kwargs_per_session.append(session_to_nwb_kwargs)

        return session_to_nwb_kwargs_per_session

# if __name__ == "__main__":
#     params = AnalysisParams()
#     params.validate_all_other_params()
#     pipeline = Pipeline(params)
#
#     data_dir_path = Path('/mnt/308A3DD28A3D9576/SYNGAP_ephys')
#     output_dir_path = Path('/media/prignane/data_fast/conversion_nwb')
#
#     kwargs = pipeline.extract_kwargs_NWB_conversion(data_dir_path, output_dir_path)
#     print(kwargs)
