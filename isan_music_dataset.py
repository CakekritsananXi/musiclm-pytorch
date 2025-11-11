"""
Isan Music Dataset and Preprocessing Module
สำหรับประมวลผลดนตรีอีสานแบบดั้งเดิม
พิณ แคน โหวด โปงลาง
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import librosa
from dataclasses import dataclass

@dataclass
class IsanInstrumentConfig:
    """Configuration for traditional Isan instruments"""
    name: str
    frequency_range: Tuple[float, float]  # min, max frequency in Hz
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    
# Traditional Isan instruments configuration
ISAN_INSTRUMENTS = {
    'พิณ': IsanInstrumentConfig(
        name='พิณ',
        frequency_range=(200, 2000),  # String instrument with mid-range frequencies
        sample_rate=16000
    ),
    'แคน': IsanInstrumentConfig(
        name='แคน',
        frequency_range=(80, 800),    # Mouth organ with low to mid frequencies
        sample_rate=16000
    ),
    'โหวด': IsanInstrumentConfig(
        name='โหวด',
        frequency_range=(100, 1200),  # Drum-like percussion
        sample_rate=16000
    ),
    'โปงลาง': IsanInstrumentConfig(
        name='โปงลาง',
        frequency_range=(150, 1500),  # Bamboo percussion
        sample_rate=16000
    )
}

class IsanMusicDataset(torch.utils.data.Dataset):
    """
    Dataset สำหรับดนตรีอีสาน
    Dataset for Isan traditional music generation
    """
    
    def __init__(
        self,
        data_path: str,
        instrument: str = 'พิณ',
        sample_rate: int = 16000,
        duration: float = 10.0,  # seconds
        transform=None
    ):
        self.data_path = Path(data_path)
        self.instrument = instrument
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples_per_track = int(sample_rate * duration)
        self.transform = transform
        
        if instrument not in ISAN_INSTRUMENTS:
            raise ValueError(f"Instrument {instrument} not supported. Available: {list(ISAN_INSTRUMENTS.keys())}")
        
        self.instrument_config = ISAN_INSTRUMENTS[instrument]
        self.audio_files = self._load_audio_files()
        
    def _load_audio_files(self) -> List[Path]:
        """โหลดไฟล์เสียงดนตรีอีสาน"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.data_path.rglob(f'*{ext}'))
        
        if not audio_files:
            print(f"⚠️ ไม่พบไฟล์เสียงสำหรับ {self.instrument} ใน {self.data_path}")
            print(f"สร้างไฟล์ตัวอย่างสำหรับการพัฒนา...")
            self._create_sample_audio()
            return self._load_audio_files()
            
        return audio_files
    
    def _create_sample_audio(self):
        """สร้างไฟล์เสียงตัวอย่างสำหรับการพัฒนา"""
        sample_dir = self.data_path / self.instrument
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # สร้างเสียงตัวอย่างสำหรับแต่ละเครื่องดนตรี
        duration_samples = int(self.sample_rate * 2)  # 2 วินาที
        t = np.linspace(0, 2, duration_samples)
        
        if self.instrument == 'พิณ':
            # พิณ - เสียงสายเครื่องดนตรี
            frequency = 440  # A4
            audio = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 2)
            
        elif self.instrument == 'แคน':
            # แคน - เสียงปากเป่า
            frequency = 220  # A3
            audio = np.sin(2 * np.pi * frequency * t) * 0.8
            # เพิ่มฮาร์มอนิกสำหรับเสียงปากเป่า
            audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
            audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
            
        elif self.instrument == 'โหวด':
            # โหวด - เสียงกลอง
            # สร้างจังหวะแบบอีสาน
            rhythm = np.zeros_like(t)
            beat_interval = int(self.sample_rate * 0.5)  # 0.5 วินาทีต่อจังหวะ
            for i in range(0, len(rhythm), beat_interval):
                if i < len(rhythm):
                    rhythm[i] = 1.0
            audio = rhythm * np.exp(-t * 5)  # ลดความเข้มอย่างรวดเร็ว
            
        elif self.instrument == 'โปงลาง':
            # โปงลาง - เสียงไม้ไผ่
            frequency = 330  # E4
            audio = np.sin(2 * np.pi * frequency * t) * 0.6
            # เพิ่ม resonance สำหรับเสียงไม้
            audio += 0.2 * np.sin(2 * np.pi * frequency * 1.5 * t)
        
        # Normalize and save
        audio = audio / np.max(np.abs(audio))
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        
        sample_path = sample_dir / f"{self.instrument}_sample.wav"
        torchaudio.save(sample_path, audio_tensor, self.sample_rate)
        print(f"✅ สร้างไฟล์ตัวอย่าง: {sample_path}")
    
    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """ประมวลผลเสียงเบื้องต้น"""
        # ตัดหรือเติมเสียงให้มีความยาวตามที่กำหนด
        if audio.shape[0] > self.samples_per_track:
            audio = audio[:self.samples_per_track]
        elif audio.shape[0] < self.samples_per_track:
            padding = self.samples_per_track - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        return audio
    
    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """ดึงฟีเจอร์สำหรับดนตรีอีสาน"""
        # Convert to mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.instrument_config.n_fft,
            hop_length=self.instrument_config.hop_length,
            n_mels=self.instrument_config.n_mels
        )
        
        mel_spec = mel_transform(audio)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        return mel_spec
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """รับข้อมูลเสียงและฟีเจอร์"""
        audio_path = self.audio_files[idx]
        
        # โหลดไฟล์เสียง
        audio, sr = torchaudio.load(audio_path)
        
        # แปลง sample rate ถ้าจำเป็น
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # ใช้แชนแนลแรกถ้ามีหลายแชนแนล
        if audio.shape[0] > 1:
            audio = audio[0]
        
        # ประมวลผลเบื้องต้น
        audio = self._preprocess_audio(audio)
        
        # ดึงฟีเจอร์
        features = self._extract_features(audio)
        
        # ใช้ transform ถ้ามี
        if self.transform:
            features = self.transform(features)
        
        return features, audio

class IsanMusicPreprocessor:
    """ตัวประมวลผลดนตรีอีสาน"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def apply_isan_effects(self, audio: torch.Tensor, instrument: str) -> torch.Tensor:
        """ใส่เอฟเฟกต์แบบดนตรีอีสาน"""
        if instrument == 'พิณ':
            return self._apply_phin_effects(audio)
        elif instrument == 'แคน':
            return self._apply_khaen_effects(audio)
        elif instrument == 'โหวด':
            return self._apply_woad_effects(audio)
        elif instrument == 'โปงลาง':
            return self._apply_ponglang_effects(audio)
        return audio
    
    def _apply_phin_effects(self, audio: torch.Tensor) -> torch.Tensor:
        """เอฟเฟกต์สำหรับพิณ"""
        # เพิ่ม vibrato เล็กน้อย
        t = torch.linspace(0, len(audio) / self.sample_rate, len(audio))
        vibrato = 1 + 0.1 * torch.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato
        return audio * vibrato
    
    def _apply_khaen_effects(self, audio: torch.Tensor) -> torch.Tensor:
        """เอฟเฟกต์สำหรับแคน"""
        # เพิ่ม breath noise และ resonance
        noise = torch.randn_like(audio) * 0.05
        return audio + noise
    
    def _apply_woad_effects(self, audio: torch.Tensor) -> torch.Tensor:
        """เอฟเฟกต์สำหรับโหวด"""
        # เพิ่ม reverb สำหรับเสียงกลอง
        return audio * 0.9  # ลดความเข้มเล็กน้อย
    
    def _apply_ponglang_effects(self, audio: torch.Tensor) -> torch.Tensor:
        """เอฟเฟกต์สำหรับโปงลาง"""
        # เพิ่ม harmonic enhancement
        return audio * 1.1

def create_isan_music_dataloader(
    data_path: str,
    instrument: str,
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader:
    """สร้าง DataLoader สำหรับดนตรีอีสาน"""
    dataset = IsanMusicDataset(data_path, instrument, **kwargs)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    print("🎵 เริ่มต้นการประมวลผลดนตรีอีสาน")
    
    # สร้าง dataset ตัวอย่าง
    dataset = IsanMusicDataset(
        data_path="./isan_audio_data",
        instrument="พิณ",
        duration=5.0
    )
    
    print(f"📊 จำนวนข้อมูล: {len(dataset)}")
    
    # ทดสอบโหลดข้อมูล
    if len(dataset) > 0:
        features, audio = dataset[0]
        print(f"🔊 รูปร่างฟีเจอร์: {features.shape}")
        print(f"🎶 รูปร่างเสียง: {audio.shape}")
        
    print("✅ การประมวลผลเสร็จสมบูรณ์")