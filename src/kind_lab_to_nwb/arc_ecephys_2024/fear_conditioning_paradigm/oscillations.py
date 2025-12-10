from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.coherence import CoherenceAnalyzer
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.coherogram import CoherogramExtractor
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.psd import PSDsAnalyzer
from kind_lab_to_nwb.arc_ecephys_2024.fear_conditioning_paradigm.spectrogram import SpectrogramExtractor


class OscillationsAnalyzer:
    def __init__(self, session):
        self.session = session
        self.params = session.params

    def _process(self, epochs_frequency_timing, time_frequency_timing):
        analyzers = [
            PSDsAnalyzer(self.session, epochs_frequency_timing),
            CoherenceAnalyzer(self.session, epochs_frequency_timing),
            SpectrogramExtractor(self.session, time_frequency_timing),
            CoherogramExtractor(self.session, time_frequency_timing),
        ]
        for analyzer in analyzers:
            analyzer.process()

    def process_pan_sessions(self):
        self._process(
            self.params.pan_sessions_epochs_frequency_analysis_timings,
            self.params.pan_sessions_time_frequency_analysis_timings
        )

    def process_baseline(self):
        self._process(
            self.params.baseline_epochs_frequency_analysis_timings,
            self.params.baseline_time_frequency_analysis_timings
        )

    def process_recall(self):
        self._process(
            self.params.recall_epochs_frequency_analysis_timings,
            self.params.recall_time_frequency_analysis_timings
        )

    def process_cond(self):
        self._process(
            self.params.cond_epochs_frequency_analysis_timings,
            self.params.cond_time_frequency_analysis_timings
        )
