from torch.utils.data import Dataset, DataLoader
import os,wfdb
import numpy as np
import neurokit2 as nk
from Utils import get_logger
from Dataset.Helpers import min_max_scaler
import torch
import random


class PPGDataset(Dataset):
    file_names: list = []
    raw_ppg_signals: list = []
    cleaned_ppg_signals: list = []
    ppg_idx: int = 0

    def __init__(self, folder_path:str, sampling_rate:int=125, window_size:int=250, quality_threshold:float=0.95, padding:int=1,shift:int=1):
        self.logger = get_logger()
        self.folder_path = folder_path
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.quality_threshold = quality_threshold
        self.padding = padding
        self.shift = shift
        self.file_names = list()
        self.ppg_signals = list()
        self.raw_ppg_signals = list()
        self.cleaned_ppg_signals = list()
        self.extract_data()

    def extract_data(self):
        self.find_ppg_files()
        self.logger.info(f"Found {len(self.file_names)} PPG files in {self.folder_path}")
        for file_name in self.file_names:
            self.logger.info(f"Processing file {file_name}")
            raw_signal = self.load_file(file_name)
            if raw_signal.size == 0:
                self.logger.warning(f"Skipping file {file_name} due to loading error.")
                continue
            cleaned_signal, quality = self.clean_ppg_signal(raw_signal)
            if cleaned_signal.size == 0 or quality.size == 0:
                self.logger.warning(f"Skipping file {file_name} due to cleaning error.")
                continue
            cleaned_segments, raw_segments, num_segments = self.extract_high_quality_segments(cleaned_signal, raw_signal, quality)
            self.cleaned_ppg_signals.extend(cleaned_segments)
            self.raw_ppg_signals.extend(raw_segments)
            self.logger.info(f"Extracted {num_segments} high-quality segments from {file_name}")
        assert len(self.cleaned_ppg_signals) == len(self.raw_ppg_signals), "Mismatch between cleaned and raw PPG segments"
        self.logger.info(f"Total high-quality PPG segments extracted: {len(self.cleaned_ppg_signals)}")

    def find_ppg_files(self):
        records = "RECORDS"
        try:
            if not os.path.exists(os.path.join(self.folder_path, records)):
                self.logger.error(f"{records} file not found in {self.folder_path}")
                raise FileNotFoundError(f"{records} file not found in {self.folder_path}")
            with open(os.path.join(self.folder_path, records), 'r') as f:
                for line in f:
                    self.file_names.append(line.strip())
        except Exception as e:
            self.logger.error(f"Error reading PPG files: {e}")
        return []
    
    def load_file(self, file_name:str)->np.ndarray:
        try:
            record = wfdb.rdrecord(os.path.join(self.folder_path, file_name))
            return record.p_signal[:, self.ppg_idx]
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return np.array([])
        
    def clean_ppg_signal(self, signal:np.ndarray)->tuple:
        try:
            signals, info = nk.ppg_process(signal, sampling_rate=self.sampling_rate)
            cleaned_signal = signals['PPG_Clean']
            quality = signals['PPG_Quality']
            return cleaned_signal, quality
        except Exception as e:
            self.logger.error(f"Error cleaning PPG signal: {e}")
            return np.array([]), np.array([])

    def extract_high_quality_segments(self, raw_signal: np.ndarray, cleaned_signal: np.ndarray, quality: np.ndarray) -> tuple:
        cleaned_segments = []
        raw_segments = []
        total_length = len(cleaned_signal)
        segment_length = self.window_size + 2 * self.padding * self.sampling_rate
        for start in range(0, total_length - segment_length, self.shift):
            segment = cleaned_signal[start:start + segment_length]
            quality_segment = quality[start:start + segment_length]
            raw_segment = raw_signal[start:start + segment_length]
            if quality_segment.min() >= self.quality_threshold:
                cleaned_segments.append(min_max_scaler(np.array(segment[self.padding*self.sampling_rate:-self.padding*self.sampling_rate])))
                raw_segments.append(min_max_scaler(np.array(raw_segment[self.padding*self.sampling_rate:-self.padding*self.sampling_rate])))
        return cleaned_segments, raw_segments, len(cleaned_segments)


    def __len__(self):
        return len(self.raw_ppg_signals)

    def __getitem__(self, idx):
        sample = torch.tensor(self.raw_ppg_signals[idx].astype(np.float32)).unsqueeze(0), torch.tensor(self.cleaned_ppg_signals[idx].astype(np.float32)).unsqueeze(0)
        return sample


    
class ContrastivePPGDataset(Dataset):
    af_file_names: list = []
    sr_file_names: list = []
    af_ppg_signals: list = []
    sr_ppg_signals: list = []
    pairs: list[tuple] = []
    ppg_idx: int = 0

    def __init__(self, folder_path:str, sampling_rate:int=125, window_size:int=250, quality_threshold:float=0.95, padding:int=1,shift:int=1,num_pairs_per_class:int=1000):
        self.logger = get_logger()
        self.folder_path = folder_path
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.quality_threshold = quality_threshold
        self.padding = padding
        self.shift = shift
        self.num_pairs_per_class = num_pairs_per_class
        self.af_file_names = list()
        self.sr_file_names = list()
        self.af_ppg_signals = list()
        self.sr_ppg_signals = list()
        self.pairs = []
        self.extract_data()
        self.create_pairs()

    def extract_data(self):
        self.find_ppg_files()
        self.logger.info(f"Found SR {len(self.sr_file_names)} PPG files in {self.folder_path}")
        self.logger.info(f"Found AF {len(self.af_file_names)} PPG files in {self.folder_path}")
        for file_name in self.sr_file_names[:1]:
            self.logger.info(f"Processing file {file_name}")
            sr_path = os.path.join(self.folder_path, 'SR-Data')
            raw_signal = self.load_file(sr_path, file_name)
            if raw_signal.size == 0:
                self.logger.warning(f"Skipping file {file_name} due to loading error.")
                continue
            cleaned_signal, quality = self.clean_ppg_signal(raw_signal)
            if cleaned_signal.size == 0 or quality.size == 0:
                self.logger.warning(f"Skipping file {file_name} due to cleaning error.")
                continue
            cleaned_segments, num_segments = self.extract_high_quality_segments(cleaned_signal, raw_signal, quality, self.quality_threshold)
            self.af_ppg_signals.extend(cleaned_segments)
            self.logger.info(f"Extracted {num_segments} high-quality segments from {file_name}")
        for file_name in self.af_file_names[:1]:
            self.logger.info(f"Processing file {file_name}")
            af_path = os.path.join(self.folder_path, 'AF-Data')
            raw_signal = self.load_file(af_path, file_name)
            if raw_signal.size == 0:
                self.logger.warning(f"Skipping file {file_name} due to loading error.")
                continue
            cleaned_signal, quality = self.clean_ppg_signal(raw_signal)
            if cleaned_signal.size == 0 or quality.size == 0:
                self.logger.warning(f"Skipping file {file_name} due to cleaning error.")
                continue
            cleaned_segments, num_segments = self.extract_high_quality_segments(cleaned_signal, raw_signal, quality, self.quality_threshold)
            self.sr_ppg_signals.extend(cleaned_segments)
            self.logger.info(f"Extracted {num_segments} high-quality segments from {file_name}")
        self.logger.info(f"Total high-quality PPG segments extracted: {len(self.af_ppg_signals)} from SR and {len(self.sr_ppg_signals)} from AF")

    def find_ppg_files(self):
        sr_path = os.path.join(self.folder_path, 'SR-Data')
        af_path = os.path.join(self.folder_path, 'AF-Data')
        records = "RECORDS"
        try:
            if not os.path.exists(os.path.join(af_path, records)):
                self.logger.error(f"{records} file not found in {af_path}")
                raise FileNotFoundError(f"{records} file not found in {af_path}")
            with open(os.path.join(af_path, records), 'r') as f:
                for line in f:
                    self.af_file_names.append(line.strip())
            if not os.path.exists(os.path.join(sr_path, records)):
                self.logger.error(f"{records} file not found in {sr_path}")
                raise FileNotFoundError(f"{records} file not found in {sr_path}")
            with open(os.path.join(sr_path, records), 'r') as f:
                for line in f:
                    self.sr_file_names.append(line.strip())
        except Exception as e:
            self.logger.error(f"Error reading PPG files: {e}")
        return []

    def load_file(self, folder: str, file_name: str) -> np.ndarray:
        try:
            record = wfdb.rdrecord(os.path.join(folder, file_name))
            return record.p_signal[:, self.ppg_idx]
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return np.array([])
        
    def clean_ppg_signal(self, signal:np.ndarray)->tuple:
        try:
            signals, info = nk.ppg_process(signal, sampling_rate=self.sampling_rate)
            cleaned_signal = signals['PPG_Clean']
            quality = signals['PPG_Quality']
            return cleaned_signal, quality
        except Exception as e:
            self.logger.error(f"Error cleaning PPG signal: {e}")
            return np.array([]), np.array([])

    def extract_high_quality_segments(self, raw_signal: np.ndarray, cleaned_signal: np.ndarray, quality: np.ndarray, quality_th) -> tuple:
        cleaned_segments = []
        total_length = len(cleaned_signal)
        segment_length = self.window_size + 2 * self.padding * self.sampling_rate
        for start in range(0, total_length - segment_length, self.shift):
            segment = cleaned_signal[start:start + segment_length]
            quality_segment = quality[start:start + segment_length]
            if quality_segment.min() >= quality_th:
                cleaned_segments.append(min_max_scaler(np.array(segment[self.padding*self.sampling_rate:-self.padding*self.sampling_rate])))
        return cleaned_segments, len(cleaned_segments)
    
    def create_pairs(self):
        # Generate positive pairs
        for _ in range(self.num_pairs_per_class):
            # SR positive pair
            s1, s2 = random.sample(self.sr_ppg_signals, 2)
            self.pairs.append((s1, s2, 1))
            # AFib positive pair
            a1, a2 = random.sample(self.af_ppg_signals, 2)
            self.pairs.append((a1, a2, 1))
            
            # Negative pair: SR vs AFib
            s = random.choice(self.sr_ppg_signals)
            a = random.choice(self.af_ppg_signals)
            self.pairs.append((s, a, 0))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1, s2, label = self.pairs[idx]
        s1 = torch.tensor(s1.astype('float32')).unsqueeze(0)
        s2 = torch.tensor(s2.astype('float32')).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)
        return s1, s2, label