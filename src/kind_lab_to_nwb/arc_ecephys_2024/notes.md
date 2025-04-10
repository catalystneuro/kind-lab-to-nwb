# Conversion notes for ecephys project 

## Fear Conditioning Experiment Description

### 1. Experimental Procedure
On experimental day 1, Local Field Potentials (LFP), Electro-EncephaloGram (EEG) and head-mounted accelerometer recordings were made while all rats were habituated to the testing context for 5 min. On experimental day 2, in a subset of animals (17 out of 31 animals), recordings were made for 1-3 hours. This recording session allowed the number of seizures per rat to be calculated for correlation with subsequent behavioural measures. On experimental day 3, recordings were made while all rats were habituated to the testing context in the same manner as for experimental day 1. On experimental day 4, recordings were made while a subset of 19 rats were pre-exposed to 10s of blue flashing light (5 Hz 110 lux flashes, 50 / 50 duty cycle) of the same properties as that used for subsequent conditioning and recall, the purpose of this exposure was to test for behavioural and LFP responses to the sensory stimuli prior to association with a footshock. Pre-exposure was followed immediately by conditioning. Conditioning occurred over a 21 min period and consisted of a 3 min period to allow for exploration of the chamber followed by 6 pairings of a conditioned stimulus (CS) co-terminating with the unconditioned stimulus (US). In a subset of 13 animals recordings were made in that session. The CS was a 10 sec blue flashing light (5 Hz 110 lux flashes, 50 / 50 duty cycle); the US was a 1 sec, 0.8 mA scrambled foot shock delivered through the bars of the floor; CS presentations started at 180, 360, 490, 770, 980, and 1280 sec into the training period (Katsenavaki et al 2024). 24 hr after conditioning, on experimental day 5, retention of the conditioned response was tested. Recordings were made while rats were placed into the testing context, with a 2 min period to allow for exploration, then ten 30 sec long presentations of the CS, separated by 30 sec of CS were given. A video camera mounted above or to the side of each context recorded the sessions. 

### 2. Subjects
This study used male rats (n=31, 16 wild-types and 15 Syngap+/∆-GAP) bred in-house and maintained in standard housing on a 12h/12h light dark with ad libitum access to water and food. Experimental animals were weaned from their dams postnatal day 22 (P22) and housed with their WT littermates, 2–4 animals per cage. Subjects were Evans-SGem2/PWC, hereafter referred to as Syngap+/∆-GAP (described in Katsenavaki et al., 2024). were genotyped by PCR. Genetically modified males and WT littermates ranging from 3 to 6 months of age were used for all experiments.

### 3. Data streams: Ephys + behavior
* Local Field Potential (LFP) recordings are sampled at 2 kHz, and bandpass-filtered between 0.1–600 Hz.
* Three-axis head-mounted accelerometer recordings are sampled at 500 Hz.
* TTL events are sampled at 2 kHz, recording the onsets and offsets of LED pulse events in channels 1 and 0 respectively.
* Video recordings are sampled at ~30 Hz

### 4. Devices:
* Experimental box for context habituation and fear response recall testing is a modified Coulbourne Instruments rat Habitest box dimensions 30 cm × 25 cm × 32 cm, containing a curved plastic black and white striped wall insert, smooth plastic grey floor, no electrified grid, scented with 70% ethanol by cleaning between trials.
* Experimental box for Seizure screening  is a perspex box containing familiar bedding material.
* Experimental box for fear conditioning is a standard, unmodified Habitest rat box with aluminium wall inserts and electrified shock floor (Coulbourne H10-11R-TC-SF) cleaned with Distel TM disinfectant wipes between trials.
* LFP recordings were made using a 16-channel digitizing head stage C3334 (Intan Technologies, USA) with an integrated accelerometer using OpenEphys software. 
* Behavioural paradigm triggers (CS/US) were programmed with Freeze Frame software (Actimetrics).
* Video recordings were captured using either Debut (NCH software) or OBS Studio (GPL-2.0-or-later open-source licence) softwares.

### 5. Temporal Alignment
Time synchronization between the electrophysiology, the accelerometer signals and the CS presentations are done using the TTL events timepoints of each LED pulse, aligned in real time to the data streams. Time synchronization between video recording and electrophysiology is done by aligning the first video frame where the CS is presented to the first TTL pulse event.

### 6. Additional Contextual Information
The data analysis for this project is still a work in progress, github repository and preprint or publication will be added when ready.

### Referenced paper : 
Katsanevaki, D., Till, S. M., Buller-Peralta, I., Nawaz, M. S., Louros, S. R., Kapgal, V., Tiwari, S., Walsh, D., Anstey, N. J., Petrović, N. G., Cormack, A., Salazar-Sanchez, V., Harris, A., Farnworth-Rowson, W., Sutherland, A., Watson, T. C., Dimitrov, S., Jackson, A. D., Arkell, D., Biswal, S., … Kind, P. C. (2024). Key roles of C2/GAP domains in SYNGAP1-related pathophysiology. Cell reports, 43(9), 114733. https://doi.org/10.1016/j.celrep.2024.114733

## Neurodatatype for spyglass compatibility

### Spyglass

Spyglass is a data management and analysis framework that uses DataJoint for database management and provides tools for working with neuroscience data.

#### Setup Spyglass 

To use Spyglass with this dataset, follow the standard Spyglass installation [instructions](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/#installation) with one important modification. Instead of cloning the main Spyglass repository, use the following command:
``` bash
git clone -b populate_sensor_data https://github.com/alessandratrapani/spyglass.git
```

This custom branch contains essential modifications to accommodate the specific requirements of this dataset, particularly for storing and processing accelerometer data from the Intan C3334 headstage in the SensorData table.

After cloning, continue with the standard Spyglass setup process as documented in the installation guide.

#### Intan C3334 Representation (One Probe, One Shank, 16 Electrodes)

**Conceptual Adaptation:** The [ndx-franklab-novela](https://github.com/nwb-extensions/ndx-franklab-novela-record) extension was originally designed with penetrating neural probes in mind, where "shanks" are physical projections containing multiple electrodes. For an EEG system like the Intan C3334, we're adapting this model to represent a fundamentally different recording approach.
**Probe and Shank Objects:** For this EEG headset, using a single Probe object (representing the entire headset) and a single Shank object (representing the complete electrode array) is most appropriate because:

* The physical EEG headset has no penetrating components
* All electrodes are part of one integrated system on the scalp surface
* Electrodes share a common reference coordinate system

However, the "Shank" object in the ndx-franklab-novela extension is somewhat misaligned with EEG technology. In EEG, there are no penetrating components - all electrodes sit on the scalp surface and the shank concept implies a physical relationship between electrodes that doesn't exist in the same way for EEG

**Coordinate System:** In our representation, we use a single coordinate system for all electrodes:

* x=0 at midline
* y=0 at the central line
* z=0 at scalp level
All electrodes reference this same system, reinforcing why a single shank makes sense

#### NB:
The project includes mock data generators and testing utilities:

* Mock NWB Files: Created for testing Spyglass compatibility without real data
* Validation Functions: Tests to ensure data integrity between NWB files and Spyglass database