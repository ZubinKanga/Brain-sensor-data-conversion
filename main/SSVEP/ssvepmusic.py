from ndf_lsl import ndf_lsl
import numpy as np
from sklearn.cross_decomposition import CCA
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse
from signal import signal, SIGINT
from sys import exit
from scipy import signal as scipysignal
import matplotlib.pyplot as plt
import json
import time as tm

# Hardcoded params
StimuliFilePath = "C:/Users/zkangabci/Git/ssvepon/SSVEPStimuli_Data/StreamingAssets/stimuli.json"
ConfigFilePath = "C:/Users/zkangabci/Git/ssvepon/SSVEPStimuli_Data/StreamingAssets/config.json"
#StimuliFilePath = "/home/sperdikis/Git/UoE/SSVEPOnline/SSVEPOnline/Assets/StreamingAssets/stimuli.json"
#ConfigFilePath = "/home/sperdikis/Git/UoE/SSVEPOnline/SSVEPOnline/Assets/StreamingAssets/config.json"

# Plotting params
DoPlot = False
PlotChInd = 7
myticks = np.arange(0.0, 256.0, 0.5)
myticks = myticks[4:66]

# Function for graceful exit on SIGINT
def sigint_handler(signal_received, frame):
    # Handle any cleanup here. Note that lsl needs no exit calls
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    osc_terminate()
    exit(0)

# SSVEP stimuli params loaded from json common with Unity protocol
jsonStimuliFile = open(StimuliFilePath, "r")
jsonContent = jsonStimuliFile.read()
Stimuli = json.loads(jsonContent)

# BCI config params loaded from json )same place as Unity protocol for consistency)
jsonConfigFile = open(ConfigFilePath, "r")
jsonConfig = jsonConfigFile.read()
BCIConfig = json.loads(jsonConfig)

basefreqs = np.zeros(len(Stimuli['Stimuli']))
for fr in range(len(Stimuli['Stimuli'])):
    # Base frequencies existing
    basefreqs[fr] = Stimuli['Stimuli'][fr]['Frequency']

# Define how many harmonics to use, NHarmonic = 1 means only base frequency is used
NHarmonic = BCIConfig['CCA']['NHarmonics']

# Define channels to use
#ChannelInd = list(range(3,9)) # Keep occipital only
#ChannelInd = list(range(6,9)) # Keep O1, Oz, O2 only
ChannelInd = BCIConfig['CCA']['ChannelInd'] # Load from config
# Convert to 0-indexing
ChannelInd = [x-1 for x in ChannelInd]

# Create NDF structure to read EEG data
ndf = ndf_lsl(stream_type="EEG", buffersize=BCIConfig['BufferSize'], frame_rate=BCIConfig['FrameRate'])
ndf.ndf_setup()

# Generate ideal oscillations for CCA algorithm
# Use sin and cos for each of the first NHarmonic
targetsignals = [] # keep as list first, makes appending easier
time = np.arange(0, float(ndf.buffer_size_samples)/float(ndf.sampling_rate), step=1.0/ndf.sampling_rate)
for fr in range(len(basefreqs)):
    for harm in range(NHarmonic):
        targetsignals.append(np.sin(2*np.pi*(harm+1)*basefreqs[fr]*time)) # add sin
        targetsignals.append(np.cos(2*np.pi*(harm+1)*basefreqs[fr]*time))  # add cos
targetsignals = np.array(targetsignals) # convert to numpy array for use with sklearn


# Prepare CCA object
cca = CCA(1) # single component

# Exponential smoothing integration parameter
do_integration = BCIConfig['integration']['do']
do_rejection = BCIConfig['rejection']
expsmooth = BCIConfig['integration']['expsmooth']
rejection = 1.0/len(basefreqs) + 0.05
integratedprob = np.ones(len(basefreqs))
integratedprob = integratedprob/sum(integratedprob)

# Pre-processing params
# Butterworth band-pass filter
buttord = BCIConfig['bandpass']['order']
low_cutoff = BCIConfig['bandpass']['cutoff']['low'] # Hz
high_cutoff = BCIConfig['bandpass']['cutoff']['high'] # Hz # Hz
w_l = low_cutoff/(ndf.sampling_rate/2)
w_h = high_cutoff/(ndf.sampling_rate/2)
b, a = scipysignal.butter(buttord, [w_l, w_h], btype='bandpass')

# OSC-related initializations and params
osc_startup()
# Make client that will receive packets
# This assumes the convention that the "BCI" server/host machine
# (the one running the EEG acquisition and this BCI script)
# lives on a machine with IP 10.66.1.1,and the client "feedback"
# Mac machine (the one running the Max/MSP patch, etc.) that
# should receive the OSC messages has IP 10.66.1.2. The
# two machines are in a local cabled network. The default port for
# UDP messages is 5000
host_ip = BCIConfig['network']['host_ip']
client_ip = BCIConfig['network']['client_ip']
port = BCIConfig['network']['client_port']
osc_udp_client(client_ip, port, "ssvepmusic")
osctag = "," + 'f'*len(basefreqs)

# Set exit handler
signal(SIGINT, sigint_handler)

###############################################################################
# Enter eternal man loop acquiring EEG and inferring SSVEP class
###############################################################################
# Read EEG data in frames
while True:
    #tm_s = tm.time()
    buffer = ndf.ndf_read()[1]
    eeg = buffer[:, ChannelInd]

#    # Patch for fake eeg testing
#    harm=1
#    fr=5
#    eeg = np.sin(2 * np.pi * (harm + 1) * basefreqs[fr] * time)
#    eeg = np.transpose([eeg]*8)

    if np.any(np.isnan(eeg)):
        continue # skip if buffer has nans

    # Process EEG buffer
    # Apply forward-backward band-pass filter
    eeg = scipysignal.filtfilt(b, a, eeg, axis=0)
    # Apply DC removal
    eeg = eeg - eeg.mean(axis=0, keepdims=True)
    # No sense for spatial filtering (not even CAR) with this minimal setup

    if DoPlot:
        # Extract PSDs of selected channel
        f, Pxx = scipysignal.welch(eeg[:, PlotChInd], fs=512, nperseg=1024, noverlap=512)
        plt.cla()
        plt.plot(f[4:66], Pxx[4:66], color='C0')
        #plt.axis([2, 32, 0, 50])
        #plt.xticks(myticks)
        plt.draw()
        plt.pause(0.001)

    # Inference on current buffer with CCA
    correlations = np.zeros(len(basefreqs))
    for fr in range(len(basefreqs)):
        response = targetsignals[fr*NHarmonic*2:(fr+1)*NHarmonic*2, :].T
        cca.fit(eeg, response)
        O1_a, O1_b = cca.transform(eeg, response)
        correlations[fr] = np.corrcoef(O1_a.T, O1_b.T)[0, 1]
    currentprob = correlations/sum(correlations)

    if do_rejection:
        # Rejection
        if np.max(currentprob) < rejection:
            currentprob = np.ones(len(basefreqs))/len(basefreqs)

    if do_integration:
        # Integration
        integratedprob = expsmooth*integratedprob + (1.0-expsmooth)*currentprob
        print(integratedprob)

    # Pack and send correlations as OSC messages
    #msg = oscbuildparse.OSCMessage("/probs", osctag, correlations)
    msg = oscbuildparse.OSCMessage("/probs", osctag, integratedprob)
    osc_send(msg, "ssvepmusic")
    osc_process()
    #tm_e = tm.time()
    #print("Elapsed time = ", 1000 * (tm_e - tm_s))
