import os
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt

SAMPLING_RATE = 125
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'Data')
af_dir = os.path.join(data_dir, 'Af-Data')
sr_dir = os.path.join(data_dir, 'SR-Data')
records = "RECORDS"
files = []
with open(os.path.join(af_dir, records), 'r') as f:
    for line in f:
        files.append(line.strip())

print("Files to be processed:", files)
sample_record = None
for file in files:
    record = wfdb.rdrecord(os.path.join(af_dir, file))
    p_signal = record.p_signal
    names = record.sig_name
    ppg_idx = 0
    for i, name in enumerate(names):
        if name == 'PPG':
            ppg_idx = i
            break
    ppg_signal = p_signal[:, ppg_idx]
    print(f"Processed file: {file}, PPG signal length: {len(ppg_signal)}")
    sample_record = ppg_signal


print("Sample PPG signal from the last processed record:", sample_record)
segment = sample_record[:500]
signals, info = nk.ppg_process(segment, sampling_rate=SAMPLING_RATE)
print(info)
quality = nk.ppg_analyze(signals, sampling_rate=SAMPLING_RATE)
