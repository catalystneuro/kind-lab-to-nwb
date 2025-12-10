%%% TAINI FILES SPIKE AND WAWE DISCHARGES (SWD) AUTODETECTION
%%% DETECTS ABSCENCE SEIZURES BY CEPSTRUM ANALYSIS OF SIGNAL
%%% Ingrid Buller UoE 2021- ingrid.buller@ed.ac.uk

function TAINI_autodetect(cn1,pn,pna,pno,day,ratname,line,T,k,rat_index,Num_index)
         
%% DEFINE RECORDING (FOR TXT FILES)
%%% VARIABLES TO FEED
%pn='C:\Users\ibull\Documents\POSTDOC\ANALISYS\EEG RESULTS\For Igor\S7064'; %% Main folder
%d='S7064-Drug2-ok'; %% Recording folder
%day= 'Drug2'; %% Day
%ratname='S7064'; %% Rat name
%A = 'eeg_OK2.txt'; % Channel to Analyse (CH3 BETTER (taini 4)!)
%nombre = [ratname '_' day];
%c=[nombre '_' A]; % Channel to Analyse (CH3 BETTER!)

%% DEFINE THRESHOLD CORRECTION FOR TAINI
Taini_correction = 1; % 1 for Taini, 0 for Open Ephys

%% DEFINE WHOLE OR PART OF RECORDING
Define_sample = 1; %% if 0 uses whole recording, if 1 to use defined start/end below
start_t=1; %% Sample point to start analysis 
end_t=21634560; %% Sample point to end analysis

%% DEFINE SAMPLING RATE
Origin_rate = 250 %% Original Sampling Rate in Hz
Real_rate = 250.4 %% Real TAINI Sampling Rate in Hz
Downsamp_factor = 1 %% To get sampling rate at 1 KHz (better analysis). i.e = for 250 original, upsampling factor = 4 to get 1 KHz

%% FOR TXT FILES
%c1= [dn '\' c]; 
%samprate = Origin_rate/Downsamp_factor;
%%% Load channel as Mat
%cn1=dlmread(c1);

%%% Subsample to part of recording
if Define_sample == 1
	samp_start = start_t/Downsamp_factor;
      else
	samp_start = 1;
end

if Define_sample == 1
	samp_end = end_t/Downsamp_factor;
      else
	samp_end = length(cn1);
end    

%%% Downsampling
ch1=downsample(cn1,Downsamp_factor);

%%% DC component removal
dcoff_1=ch1-mean(ch1);

%%% Remove-Fill outliers
outlier_off = filloutliers(dcoff_1,'linear','ThresholdFactor',4.5);

%%% Highpass
ch_filt = highpass(outlier_off,0.02,'Steepness',0.5,'StopbandAttenuation',120);

%%%CENTER VALUES ON ZERO AND REMOVE NOISE
%middle=median(ch_filt);
%centered=ch_filt-middle;
%eegch = [];
%for i = 1:numel(ch_filt)
%	if ch_filt(i) >1 | ch_filt(i) <-1
%	   	   eegch(i) = 0;
%	        else
%	           eegch(i) = ch_filt(i)*1000; 
%	        end
%end

eegch = ch_filt*1000;

if Real_rate==250.4
    eegch(1252:1252:end,:)=[];
    eegch(1251:1251:end,:)=[];
else
    eegch=eegch;
end    

eegch=eegch';
samprate=250;
cepstral_analysis(eegch,samprate,ratname,day,line,pn,pna,pno,Taini_correction,T,k,rat_index,Num_index)

