%%% TAINI FILES SPIKE AND WAWE DISCHARGES (SWD) AUTODETECTION
%%% DETECTS ABSCENCE SEIZURES BY CEPSTRUM ANALYSIS OF SIGNAL
%%% Ingrid Buller UoE 2021- ingrid.buller@ed.ac.uk
function cepstral_analysis(eegch,samprate,ratname,day,line,pn,pna,pno,Taini_correction,T,k,rat_index,Num_index)


tstartstop = [0 length(eegch)/samprate]; %In secs
done=1;
while done==1
[spec]=first_spec(eegch,tstartstop,samprate);
done = exist('spec')-1;
end
done=1;
while done==1
	timedomain = (1:1:length(eegch));
	freqdomain = getfield(spec,'S');
	logspec = log(abs(freqdomain)+sqrt(-1)*unwrap(angle(freqdomain)));
	done=0;
end

%%% WAIT HERE FOR SPECTROGRAM COMPUTATION------------------------------------------------------------
logspec = logspec'; % rows to cols

%%%--------------------------------------------------------------------------------------------------
%% CEPSTRAL ANALYSIS--------------------------------------------------------

cspectral = real(ifft(logspec)); %cepstrum analisys
freq = getfield(spec,'f'); %freq vector in hz
tiempo = getfield(spec,'t'); %time vector in secs
yt=[,0.10 0.30 ];
yt=yt';
tiempo_fin=length(tiempo);
yy=[, tiempo(1,tiempo_fin)+0.2,tiempo(1,tiempo_fin)+0.4,tiempo(1,tiempo_fin)+0.6];
yy=yy';
tiempo = tiempo';
timesec =[yt;tiempo;yy];
pw_cepts = 4*(cspectral);

bins = samprate./freq;
q = bins/samprate; % Quefrency
q(~isfinite(q)) = 2*q(1,2);
half_q = 1./q;

%%%%THRESHOLD ON THETA BAND OR LOWEST-------------------------------
if Taini_correction == 1
    delta=sum(pw_cepts(3:12,:)); %(eliminate delta from main component)
    %filt_cespt=pw_cepts(1,:)-delta;
    %theta_cespt=pw_cepts([16:164],:);
    theta_cespt=pw_cepts(1,:)-delta;
    th=(16);
    %th=(0.8);
    %maxval = max(theta_cespt,[],1); %max value in theta
    %peak=max(maxval); %max theta value in all bins
    %seizscore=maxval/peak; %proportion of max
    %seizscore(isnan(seizscore))=0; %Clears NANs
    %seiz_cepstral=seizscore;
    seiz_cepstral=theta_cespt;
    plotcheck = seiz_cepstral';
else
    theta_cespt = pw_cepts([10:20],:); %(focus on theta)
    th=(2.2*10^-5);
    maxval = max(theta_cespt,[],1); %max value in theta
    peak=max(maxval); %max theta value in all bins
    seizscore=maxval/peak; %proportion of max
    seizscore(isnan(seizscore))=0; %Clears NANs
    norm_seizscore = normalize(seizscore); %normalize values to homogenize amplitude differences
    seiz_cepstral=norm_seizscore;
end

seiz_cepstral(seiz_cepstral < th) = 0; %sets 1 to seizure bins (2.2 SD from avg)
seiz_cepstral(seiz_cepstral >= th) = 1; %sets 1 to seizure bins (2.2 SD from avg)
id_cspectral = pw_cepts;
[~,n] = size(logspec);

%%%--------------------------------------------------------------------------------------------------
%% REMOVING NOISE TO AVOID THRESHOLD MISTAKES--------------------------------------------------------
%[m,n] = size(logspec);
%speclog_unoise = zeros(m,n);

%for i=1:n
%	if   logspec(45,i) >= logspec(62,i) | logspec(20,i) >= logspec(32,i) | logspec(291,i) >= logspec(45,i) | logspec(1,i) >= 10 | logspec(328,i) >= 2 | logspec(32,i) < 5
%	   	   speclog_unoise(:,i) = 0;
%	        else
%	           speclog_unoise(:,i) = logspec(:,i);
%	        end
%end

%cspectral = real(ifft(speclog_unoise)); %cepstrum analisys
%maxval = max(cspectral,[],1); %max value of each bin
%timesec = getfield(spec,'t'); %time vector in secs
%seiz=max(maxval); %max value in all bins
%freq = getfield(spec,'f'); %freq vector in hz
%seizscore=maxval/seiz; %proportion of max

%id_cspectral = seizscore;
%seiz_cepstral=id_cspectral;
%seiz_cepstral(seiz_cepstral >= 0.7) = 1; %sets 1 to seizure bins (70% of max value)

%%%--------------------------------------------------------------------------------------------------
%%% GET DGE SEIZ DGE FOR IGOR COMPARISON-------------------------------------------------------------
DGE_seiz = [];
for i = 1:numel(seiz_cepstral)
	if seiz_cepstral(i) == 1 
	   	   DGE_seiz(i) = 6;
	        else
	           DGE_seiz(i) = 4;
	        end
end
xt=[,4 4 ];
xt=xt';
xx=[, 4 4 4];
xx=xx';
DGE_seiz = DGE_seiz'; %TO COLUMN
Seiz_score=[xt;DGE_seiz;xx];

%%%--------------------------------------------------------------------------------------------------
%%%GET TIME START-STOP AND TOTALS -------------------------------------------------------------------
%timesec=timesec';
score_start=[];
    for i = 2:n
            if Seiz_score(i) ==  6 & Seiz_score(i-1) ==  4 
		     score_start(i) = 1;
               else
		     score_start(i) = 0;
	    end
    end
score_start=score_start';
times=find(score_start>0);
time_start=timesec(times);

score_end=[];
    for i = 1:n
            if Seiz_score(i) ==  6 & Seiz_score(i+1) ==  4 | Seiz_score(end) ==  6
		     score_end(i) = 1;
               else
		     score_end(i) = 0;
	    end
    end
score_end=score_end';
times=find(score_end>0);
time_end=timesec(times);

 for i = 1:numel(time_start)
            if Seiz_score(i) ==  6 & Seiz_score(i+1) ==  4 | Seiz_score(end) ==  6
		     score_end(i) = 1;
               else
		     score_end(i) = 0;
	    end
    end 
%%%--------------------------------------------------------------------------------------------------
%%%CHECK IF START-END VECTOR HAVE SAME SIZE ---------------------------------------------------------
if numel(time_end(:,1)) < numel(time_start(:,1))
   time_start(end,:) = []; 
end

if numel(time_end(:,1)) > numel(time_start(:,1))
   time_end(end,:) = []; 
end

dur=time_end-time_start;
if max(Seiz_score) == 4
    Num_index(k,5)=0;
    new_tab=array2table(Num_index);
    T(:,9:15)=new_tab;
    cd(pno);
    outputdir=pwd;
    writetable(T,['rat_day_index_' line '.csv'])
    disp 'NO SEIZURES FOUND!!!!!!!';
    Number_SWD=0
   return
end
 %  choice = menu('No Seizures Found Quit?','Yes','No');
%	  if choice==1 | choice==0 
 %        disp 'GO';
   %	     return;
  %    end
   %   if choice==2 | choice==0
    %            close
    %            return;
     % end
%end


%%%--------------------------------------------------------------------------------------------------
%%% (NOT!) REMOVE EVENTS WITH DURATION = 0 -----------------------------------------------------------------
ceros=find(dur>=0);
timestart_end = [time_start(ceros) time_end(ceros)]; 
if isempty(dur)
    Num_index(k,5)=0;
    new_tab=array2table(Num_index);
    T(:,9:15)=new_tab;
    cd(pno);
    outputdir=pwd;
    writetable(T,['rat_day_index_' line '.csv'])
   disp 'NO SEIZURES FOUND!!!!!!!';
   Number_SWD=0
   return
end
%%%--------------------------------------------------------------------------------------------------
%%% COLLAPSE EVENTS SEPARATED BY LESS THAN 2 SEC ----------------------------------------------------
maxLag = 2;
[idx,idx] = sort(timestart_end(1,:));
times_close = timestart_end(:,idx);
test = times_close(2:end,1)-times_close(1:end-1,2);
tooCloseIDs = find(test < maxLag );%& test >=0);

%%% COLLAPSE LOOP -----------------------------------------------------------------------------------
while numel(tooCloseIDs) > 0
    for i = numel(tooCloseIDs):-1:1
        times_close(tooCloseIDs(i),2) = times_close(tooCloseIDs(i)+1,2);
    end
    times_close(tooCloseIDs+1,:) = [];
    test = times_close(2:end,1)-times_close(1:end-1,2);
    tooCloseIDs = find(test < maxLag );%& test >=0);
    disp 'collapsing events...';
end

Falses=find((times_close(:,2)-times_close(:,1))>0.8); %%% REMOVE ISOLATED "EVENTS" OF <0.8 SEC
seizEvtsMerged= [times_close(Falses,1) times_close(Falses,2)]; 


%%%--------------------------------------------------------------------------------------------------
%%% GETS NEW DURATIONS ------------------------------------------------------------------------------
seizEvtsMerged(:,3) = seizEvtsMerged(:,2) - seizEvtsMerged(:,1);  %Gets new Dur

if length(seizEvtsMerged) <= 2 & max(seizEvtsMerged(:,3))< 5 %%% CONSIDER OVER 5 EVENTS AS SEIZURE PRESENT
   disp 'NO SEIZURES FOUND!!!!!!!';
   Num_index(k,5)=0;
   new_tab=array2table(Num_index);
   T(:,9:15)=new_tab;
   cd(pno);
   outputdir=pwd;
   writetable(T,['rat_day_index_' line '.csv'])
else
   Num_index(k,5)=1;
   new_tab=array2table(Num_index);
   T(:,9:15)=new_tab;
   cd(pno);
   outputdir=pwd;
   writetable(T,['rat_day_index_' line '.csv'])
end


%%%--------------------------------------------------------------------------------------------------
%%% GET TOTAL NUMBER OF EVENTS AND MEAN DURATION IN SECS --------------------------------------------
N_seizA=length(seizEvtsMerged(:,3));
N_seiz=[N_seizA 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
N_seiz=N_seiz';
Mean_durA=mean(seizEvtsMerged(:,3));
Mean_dur=[Mean_durA 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
Mean_dur=Mean_dur';

%%%--------------------------------------------------------------------------------------------------
%%%GET FILE FOR IGOR---------------------------------------------------------------------------------
[tf,idx] = ismember(seizEvtsMerged(:,1),timesec);
ind_start=idx;
[tf,idx] = ismember(seizEvtsMerged(:,2),timesec);
ind_end=idx;

IGOR = [];
[x,y] = size(Seiz_score);
IGOR = zeros(x,y);
for i = 1:numel(ind_start)
    IGOR(ind_start(i):ind_end(i)) = 1;
end

DGE_seiz = [];
for i = 1:numel(IGOR)
	if IGOR(i) == 1 
	   	   DGE_seiz(i) = 6;
	        else
	           DGE_seiz(i) = 4;
	        end
end
DGE_seiz = DGE_seiz';
DGE_SWDs = downsample(DGE_seiz,25);

%%%--------------------------------------------------------------------------------------------------
%%% CALCULATE SWDs BY HOUR---- ----------------------------------------------------------------------
hr=[0:1:23];
sec_hrstart=[0:3600:82800];
sec_hrend=[3600:3600:86400];
hr=hr';
sec_hrend=sec_hrend';
sec_hrstart=sec_hrstart';
tiempos=[sec_hrstart sec_hrend];

z_0=sum(seizEvtsMerged(:,1)>tiempos(1,1) & seizEvtsMerged(:,2)<=tiempos(1,2));
z_1=sum(seizEvtsMerged(:,1)>=tiempos(2,1) & seizEvtsMerged(:,2)<=tiempos(2,2));
z_2=sum(seizEvtsMerged(:,1)>=tiempos(3,1) & seizEvtsMerged(:,2)<=tiempos(3,2));
z_3=sum(seizEvtsMerged(:,1)>=tiempos(4,1) & seizEvtsMerged(:,2)<=tiempos(4,2));
z_4=sum(seizEvtsMerged(:,1)>=tiempos(5,1) & seizEvtsMerged(:,2)<=tiempos(5,2));
z_5=sum(seizEvtsMerged(:,1)>=tiempos(6,1) & seizEvtsMerged(:,2)<=tiempos(6,2));
z_6=sum(seizEvtsMerged(:,1)>=tiempos(7,1) & seizEvtsMerged(:,2)<=tiempos(7,2));
z_7=sum(seizEvtsMerged(:,1)>=tiempos(8,1) & seizEvtsMerged(:,2)<=tiempos(8,2));
z_8=sum(seizEvtsMerged(:,1)>=tiempos(9,1) & seizEvtsMerged(:,2)<=tiempos(9,2));
z_9=sum(seizEvtsMerged(:,1)>=tiempos(10,1) & seizEvtsMerged(:,2)<=tiempos(10,2));
z_10=sum(seizEvtsMerged(:,1)>=tiempos(11,1) & seizEvtsMerged(:,2)<=tiempos(11,2));
z_11=sum(seizEvtsMerged(:,1)>=tiempos(12,1) & seizEvtsMerged(:,2)<=tiempos(12,2));
z_12=sum(seizEvtsMerged(:,1)>=tiempos(13,1) & seizEvtsMerged(:,2)<=tiempos(13,2));
z_13=sum(seizEvtsMerged(:,1)>=tiempos(14,1) & seizEvtsMerged(:,2)<=tiempos(14,2));
z_14=sum(seizEvtsMerged(:,1)>=tiempos(15,1) & seizEvtsMerged(:,2)<=tiempos(15,2));
z_15=sum(seizEvtsMerged(:,1)>=tiempos(16,1) & seizEvtsMerged(:,2)<=tiempos(16,2));
z_16=sum(seizEvtsMerged(:,1)>=tiempos(17,1) & seizEvtsMerged(:,2)<=tiempos(17,2));
z_17=sum(seizEvtsMerged(:,1)>=tiempos(18,1) & seizEvtsMerged(:,2)<=tiempos(18,2));
z_18=sum(seizEvtsMerged(:,1)>=tiempos(19,1) & seizEvtsMerged(:,2)<=tiempos(19,2));
z_19=sum(seizEvtsMerged(:,1)>=tiempos(20,1) & seizEvtsMerged(:,2)<=tiempos(20,2));
z_20=sum(seizEvtsMerged(:,1)>=tiempos(21,1) & seizEvtsMerged(:,2)<=tiempos(21,2));
z_21=sum(seizEvtsMerged(:,1)>=tiempos(22,1) & seizEvtsMerged(:,2)<=tiempos(22,2));
z_22=sum(seizEvtsMerged(:,1)>=tiempos(23,1) & seizEvtsMerged(:,2)<=tiempos(23,2));
z_23=sum(seizEvtsMerged(:,1)>=tiempos(24,1) & seizEvtsMerged(:,2)<=tiempos(24,2));

%%---Mid ZT SWDs
%%-----------------------------------------------------------------------------------------
z0_1= sum(seizEvtsMerged(:,1)>tiempos(1,1) & seizEvtsMerged(:,2)<=tiempos(2,2));  
z1_2= sum(seizEvtsMerged(:,1)>tiempos(2,1) & seizEvtsMerged(:,2)<=tiempos(3,2));  
z2_3= sum(seizEvtsMerged(:,1)>tiempos(3,1) & seizEvtsMerged(:,2)<=tiempos(4,2));
z3_4= sum(seizEvtsMerged(:,1)>tiempos(4,1) & seizEvtsMerged(:,2)<=tiempos(5,2));
z4_5= sum(seizEvtsMerged(:,1)>tiempos(5,1) & seizEvtsMerged(:,2)<=tiempos(6,2));
z5_6= sum(seizEvtsMerged(:,1)>tiempos(6,1) & seizEvtsMerged(:,2)<=tiempos(7,2));
z6_7= sum(seizEvtsMerged(:,1)>tiempos(7,1) & seizEvtsMerged(:,2)<=tiempos(8,2));
z7_8= sum(seizEvtsMerged(:,1)>tiempos(8,1) & seizEvtsMerged(:,2)<=tiempos(9,2));
z8_9= sum(seizEvtsMerged(:,1)>tiempos(9,1) & seizEvtsMerged(:,2)<=tiempos(10,2));
z9_10= sum(seizEvtsMerged(:,1)>tiempos(10,1) & seizEvtsMerged(:,2)<=tiempos(11,2));
z10_11= sum(seizEvtsMerged(:,1)>tiempos(11,1) & seizEvtsMerged(:,2)<=tiempos(12,2));
z11_12= sum(seizEvtsMerged(:,1)>tiempos(12,1) & seizEvtsMerged(:,2)<=tiempos(13,2));
z12_13= sum(seizEvtsMerged(:,1)>tiempos(13,1) & seizEvtsMerged(:,2)<=tiempos(14,2));
z13_14= sum(seizEvtsMerged(:,1)>tiempos(14,1) & seizEvtsMerged(:,2)<=tiempos(15,2));
z14_15= sum(seizEvtsMerged(:,1)>tiempos(15,1) & seizEvtsMerged(:,2)<=tiempos(16,2));
z15_16= sum(seizEvtsMerged(:,1)>tiempos(16,1) & seizEvtsMerged(:,2)<=tiempos(17,2));
z16_17= sum(seizEvtsMerged(:,1)>tiempos(17,1) & seizEvtsMerged(:,2)<=tiempos(18,2));
z17_18= sum(seizEvtsMerged(:,1)>tiempos(18,1) & seizEvtsMerged(:,2)<=tiempos(19,2));
z18_19= sum(seizEvtsMerged(:,1)>tiempos(19,1) & seizEvtsMerged(:,2)<=tiempos(20,2));
z19_20= sum(seizEvtsMerged(:,1)>tiempos(20,1) & seizEvtsMerged(:,2)<=tiempos(21,2));
z20_21= sum(seizEvtsMerged(:,1)>tiempos(21,1) & seizEvtsMerged(:,2)<=tiempos(22,2));
z21_22= sum(seizEvtsMerged(:,1)>tiempos(22,1) & seizEvtsMerged(:,2)<=tiempos(23,2));
z22_23= sum(seizEvtsMerged(:,1)>tiempos(23,1) & seizEvtsMerged(:,2)<=tiempos(24,2));

%%%GET REAL TOTALS PER ZT-----------------------------------------------
zt_0 = z_0;
zt_1 = z_1+(z0_1-(z_0+z_1));
zt_2 = z_2+(z1_2-(z_1+z_2));
zt_3 = z_3+(z2_3-(z_2+z_3));
zt_4 = z_4+(z3_4-(z_3+z_4));
zt_5 = z_5+(z4_5-(z_4+z_5));
zt_6 = z_6+(z5_6-(z_5+z_6));
zt_7 = z_7+(z6_7-(z_6+z_7));
zt_8 = z_8+(z7_8-(z_7+z_8));
zt_9 = z_9+(z8_9-(z_8+z_9));
zt_10 = z_10+(z9_10-(z_9+z_10));
zt_11 = z_11+(z10_11-(z_10+z_11));
zt_12 = z_12+(z11_12-(z_11+z_12));
zt_13 = z_13+(z12_13-(z_12+z_13));
zt_14 = z_14+(z13_14-(z_13+z_14));
zt_15 = z_15+(z14_15-(z_14+z_15));
zt_16 = z_16+(z15_16-(z_15+z_16));
zt_17 = z_17+(z16_17-(z_16+z_17));
zt_18 = z_18+(z17_18-(z_17+z_18));
zt_19 = z_19+(z18_19-(z_18+z_19));
zt_20 = z_20+(z19_20-(z_19+z_20));
zt_21 = z_21+(z20_21-(z_20+z_21));
zt_22 = z_22+(z21_22-(z_21+z_22));
zt_23 = z_23+(z22_23-(z_22+z_23));

seiz_ZT=[zt_0 zt_1 zt_2 zt_3 zt_4 zt_5 zt_6 zt_7 zt_8 zt_9 zt_10 zt_11 zt_12 zt_13 zt_14 zt_15 zt_16 zt_17 zt_18 zt_19 zt_20 zt_21 zt_22 zt_23];
seiz_ZT=seiz_ZT';
Total_dayA=sum(seiz_ZT);
Total_day=[Total_dayA 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
Total_day=Total_day';




%%%--------------------------------------------------------------------------------------------------
%%% CREATE TABLES WITH RESULTS ----------------------------------------------------------------------
colNames_A = {'N_event','mean_dur','ZT','SWDs','Day'};
colNames_B = {'sec_start','sec_end','dur'};
ALL= round(seizEvtsMerged,2);
Seizures= array2table(ALL,'VariableNames',colNames_B);
Totals = [N_seiz Mean_dur hr seiz_ZT Total_day];
Seiz_Totals = array2table(Totals,'VariableNames',colNames_A);
DGE_SWD = array2table(DGE_SWDs);

done=1;
while done==1  
	choice = menu('Check Events?','Yes','No');
	if choice==1 | choice==0
   		for i = 1:N_seiz
    		figure(1);
    		set(gcf,'position',[51.4 456.2 1408 252.8])
    		clf;
    		subplot(2,1,1)
    		plot(timedomain,eegch)
		a=ALL(i,1);
		b=ALL(i,2);
		v=a*samprate;
		u=b*samprate;
    		xlim([v u])
    		ylim([-1000 1000])
    		ylabel ('Power (db)')
    		xlabel ('sample'); 
    		subplot(2,1,2)
    		imagesc(timesec,freq,logspec)
    		colormap jet
    		set(gca, 'ydir','normal', 'clim',[0 7])
    		set(gca, 'xlim' ,[a b]); %chunk of secs
    		xlabel ('Time (sec)')
    		ylabel ('Frequency (Hz)');
            %plot(tiempo,theta_cespt)
            %set(gcf,'position',[486.6 173.8 1000.0 250.0])
            %xlim([a b])
            %ylim([10 20])
            %yline(16,'-.r','LineWidth',1);
            %yline(0.8,'-.r','LineWidth',1);
            %xlabel ('Time (sec)')
            %ylabel ('f0 epoch/Max f0 (zScore)');
            %title ('Fundamental Frequency Power Normalized to Max'); 
    		choice = menu('Keep checking?','Yes','No');
	  		if choice==2 | choice==0
                close
                done=0; 
                break
	  		end
		end
    else
        close
        done=0; 
        break
    end
end

done=1;
while done==1
	choice = menu('Save Results?','Yes','No');
	if choice==1 | choice==0
   		%%% CREATE NEW DIR AND SAVE TOTALS AS .CSV ----------------------------------------------------------
		pnd =  [pna '\' 'seiz'];
        if ~exist(pnd, 'dir')
        mkdir(pnd)
        end
        cd(pnd);
		outputdir=pwd;
		writetable(Seiz_Totals,[ratname '_' day '_Seiz_Totals.csv'])
        writetable(Seizures,[ratname '_' day '_Seizures.csv'])
        writetable(DGE_SWD,[ratname '_' day '_DGE_SWDs.csv'])
        %dlmwrite([ratname '_' day '_DGE_seiz.txt'],DGE_seiz,'delimiter','\t','precision',1);
        %writetable(Channels,[ratname '_' day '_Channels.csv'])
        Num_index(k,5)=1;
           new_tab=array2table(Num_index);
           T(:,9:15)=new_tab;
           cd(pno);
           outputdir=pwd;
           writetable(T,['rat_day_index_' line '.csv'])
        done=0;
    else
           Num_index(k,5)=0;
           new_tab=array2table(Num_index);
           T(:,9:15)=new_tab;
           cd(pno);
           outputdir=pwd;
           writetable(T,['rat_day_index_' line '.csv'])
            close
            done=0;
            break
    end
done=0;
end
Number_SWD=N_seizA