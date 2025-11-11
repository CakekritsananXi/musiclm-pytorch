"""
Traditional Thai Isan Instrument Models
โมเดลสำหรับเครื่องดนตรีไทยอีสานแบบดั้งเดิม
พิณ, แคน, โหวด, โปงลาง
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class InstrumentCharacteristics:
    """ลักษณะเฉพาะของเครื่องดนตรี"""
    name: str
    frequency_range: Tuple[float, float]
    harmonic_profile: List[float]
    envelope_params: Dict[str, float]
    playing_technique: str
    cultural_context: str

# ลักษณะเฉพาะของเครื่องดนตรีอีสานแต่ละชนิด
INSTRUMENT_CHARACTERISTICS = {
    'พิณ': InstrumentCharacteristics(
        name='พิณ',
        frequency_range=(200, 2000),
        harmonic_profile=[1.0, 0.6, 0.3, 0.15, 0.08],  # ลำดับฮาร์มอนิก
        envelope_params={
            'attack': 0.05,
            'decay': 0.3,
            'sustain': 0.7,
            'release': 1.2
        },
        playing_technique='plucked_string',
        cultural_context='lead_melody_instrument'
    ),
    'แคน': InstrumentCharacteristics(
        name='แคน',
        frequency_range=(80, 800),
        harmonic_profile=[1.0, 0.8, 0.6, 0.4, 0.3, 0.2],  # Rich harmonics for mouth organ
        envelope_params={
            'attack': 0.1,
            'decay': 0.2,
            'sustain': 0.8,
            'release': 0.8
        },
        playing_technique='free_reed_aerophone',
        cultural_context='harmonic_background_instrument'
    ),
    'โหวด': InstrumentCharacteristics(
        name='โหวด',
        frequency_range=(100, 1200),
        harmonic_profile=[1.0, 0.3, 0.1],  # Percussive with few harmonics
        envelope_params={
            'attack': 0.01,
            'decay': 0.1,
            'sustain': 0.2,
            'release': 0.3
        },
        playing_technique='membranophone_percussion',
        cultural_context='rhythmic_accompaniment'
    ),
    'โปงลาง': InstrumentCharacteristics(
        name='โปงลาง',
        frequency_range=(150, 1500),
        harmonic_profile=[1.0, 0.7, 0.5, 0.3],  # Bamboo tube harmonics
        envelope_params={
            'attack': 0.02,
            'decay': 0.15,
            'sustain': 0.6,
            'release': 0.5
        },
        playing_technique='idiophone_percussion',
        cultural_context='melodic_percussion'
    )
}

class TraditionalInstrumentSynthesizer(nn.Module):
    """โมเดลสำหรับสังเคราะห์เสียงเครื่องดนตรีไทย"""
    
    def __init__(self, instrument_name: str, sample_rate: int = 16000):
        super().__init__()
        self.instrument_name = instrument_name
        self.sample_rate = sample_rate
        self.characteristics = INSTRUMENT_CHARACTERISTICS[instrument_name]
        
        # Envelope generator
        self.envelope = ADSREnvelope(**self.characteristics.envelope_params)
        
        # Harmonic generator
        self.harmonic_gen = HarmonicGenerator(
            harmonic_profile=self.characteristics.harmonic_profile
        )
        
        # Instrument-specific processor
        if instrument_name == 'พิณ':
            self.processor = PhinProcessor(sample_rate)
        elif instrument_name == 'แคน':
            self.processor = KhaenProcessor(sample_rate)
        elif instrument_name == 'โหวด':
            self.processor = WoadProcessor(sample_rate)
        elif instrument_name == 'โปงลาง':
            self.processor = PonglangProcessor(sample_rate)
    
    def forward(self, pitch: float, duration: float, velocity: float = 0.8) -> torch.Tensor:
        """สังเคราะห์เสียงเครื่องดนตรี"""
        # Generate time vector
        n_samples = int(duration * self.sample_rate)
        t = torch.linspace(0, duration, n_samples)
        
        # Generate envelope
        env = self.envelope(n_samples, self.sample_rate)
        
        # Generate harmonics
        signal = self.harmonic_gen(pitch, t)
        
        # Apply envelope
        signal = signal * env * velocity
        
        # Apply instrument-specific processing
        signal = self.processor(signal, pitch, velocity)
        
        return signal

class ADSREnvelope(nn.Module):
    """Envelope generator แบบ ADSR (Attack, Decay, Sustain, Release)"""
    
    def __init__(self, attack: float, decay: float, sustain: float, release: float):
        super().__init__()
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
    
    def forward(self, n_samples: int, sample_rate: int) -> torch.Tensor:
        t = torch.linspace(0, n_samples / sample_rate, n_samples)
        envelope = torch.zeros_like(t)
        
        # Attack phase
        attack_samples = int(self.attack * sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = torch.linspace(0, 1, attack_samples)
        
        # Decay phase
        decay_samples = int(self.decay * sample_rate)
        decay_start = attack_samples
        decay_end = decay_start + decay_samples
        if decay_samples > 0 and decay_end <= n_samples:
            envelope[decay_start:decay_end] = torch.linspace(1, self.sustain, decay_samples)
        
        # Sustain phase
        sustain_start = decay_end
        sustain_end = n_samples - int(self.release * sample_rate)
        if sustain_start < sustain_end:
            envelope[sustain_start:sustain_end] = self.sustain
        
        # Release phase
        release_samples = int(self.release * sample_rate)
        release_start = max(0, n_samples - release_samples)
        if release_samples > 0:
            envelope[release_start:] = torch.linspace(self.sustain, 0, n_samples - release_start)
        
        return envelope

class HarmonicGenerator(nn.Module):
    """Generate harmonics based on instrument profile"""
    
    def __init__(self, harmonic_profile: List[float]):
        super().__init__()
        self.harmonic_profile = harmonic_profile
        self.n_harmonics = len(harmonic_profile)
    
    def forward(self, fundamental_freq: float, t: torch.Tensor) -> torch.Tensor:
        signal = torch.zeros_like(t)
        
        for i, amplitude in enumerate(self.harmonic_profile):
            harmonic_freq = fundamental_freq * (i + 1)
            harmonic = amplitude * torch.sin(2 * np.pi * harmonic_freq * t)
            signal += harmonic
        
        return signal

class PhinProcessor(nn.Module):
    """Processor สำหรับเสียงพิณ"""
    
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate
    
    def forward(self, signal: torch.Tensor, pitch: float, velocity: float) -> torch.Tensor:
        # Add subtle vibrato
        vibrato_rate = 5.0  # Hz
        vibrato_depth = 0.02
        t = torch.linspace(0, len(signal) / self.sample_rate, len(signal))
        vibrato = 1 + vibrato_depth * torch.sin(2 * np.pi * vibrato_rate * t)
        
        signal = signal * vibrato
        
        # Add string resonance
        resonance_freq = pitch * 0.5
        resonance = 0.1 * torch.sin(2 * np.pi * resonance_freq * t)
        signal = signal + resonance
        
        return signal

class KhaenProcessor(nn.Module):
    """Processor สำหรับเสียงแคน"""
    
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate
    
    def forward(self, signal: torch.Tensor, pitch: float, velocity: float) -> torch.Tensor:
        # Add breath noise
        noise = torch.randn_like(signal) * 0.05 * (1 - velocity)
        signal = signal + noise
        
        # Add tube resonance effect
        t = torch.linspace(0, len(signal) / self.sample_rate, len(signal))
        resonance = 0.15 * torch.sin(2 * np.pi * pitch * 0.3 * t)
        signal = signal + resonance
        
        return signal

class WoadProcessor(nn.Module):
    """Processor สำหรับเสียงโหวด"""
    
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate
    
    def forward(self, signal: torch.Tensor, pitch: float, velocity: float) -> torch.Tensor:
        # Add drum-specific processing
        # Emphasize attack
        attack_enhance = torch.exp(-torch.linspace(0, 5, len(signal)))
        signal = signal * (1 + 0.3 * attack_enhance)
        
        # Add low-frequency emphasis
        # Simple low-pass effect
        signal_rolled = torch.roll(signal, 1, 0)
        signal_rolled[0] = 0
        signal = 0.7 * signal + 0.3 * signal_rolled
        
        return signal

class PonglangProcessor(nn.Module):
    """Processor สำหรับเสียงโปงลาง"""
    
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate
    
    def forward(self, signal: torch.Tensor, pitch: float, velocity: float) -> torch.Tensor:
        # Add bamboo tube resonance
        t = torch.linspace(0, len(signal) / self.sample_rate, len(signal))
        tube_resonance = 0.2 * torch.sin(2 * np.pi * pitch * 0.7 * t)
        signal = signal + tube_resonance
        
        # Add subtle metallic quality
        metallic = 0.1 * torch.sin(2 * np.pi * pitch * 2.5 * t)
        signal = signal + metallic
        
        return signal

class IsanEnsembleGenerator(nn.Module):
    """Generator สำหรับวงดนตรีอีสาน"""
    
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.instruments = nn.ModuleDict({
            name: TraditionalInstrumentSynthesizer(name, sample_rate)
            for name in INSTRUMENT_CHARACTERISTICS.keys()
        })
    
    def forward(
        self,
        composition: Dict[str, List[Dict]],
        duration: float = 10.0
    ) -> torch.Tensor:
        """
        Generate ensemble performance
        
        composition format:
        {
            'พิณ': [
                {'pitch': 440, 'start_time': 0.0, 'duration': 2.0, 'velocity': 0.8},
                ...
            ],
            'แคน': [...],
            ...
        }
        """
        n_samples = int(duration * self.sample_rate)
        ensemble_signal = torch.zeros(n_samples)
        
        for instrument_name, notes in composition.items():
            if instrument_name not in self.instruments:
                continue
            
            instrument = self.instruments[instrument_name]
            instrument_track = torch.zeros(n_samples)
            
            for note in notes:
                start_sample = int(note['start_time'] * self.sample_rate)
                note_duration = note['duration']
                
                # Generate note
                note_signal = instrument(
                    pitch=note['pitch'],
                    duration=note_duration,
                    velocity=note.get('velocity', 0.8)
                )
                
                # Add to track
                end_sample = min(start_sample + len(note_signal), n_samples)
                actual_samples = end_sample - start_sample
                if actual_samples > 0:
                    instrument_track[start_sample:end_sample] += note_signal[:actual_samples]
            
            # Mix into ensemble
            ensemble_signal += instrument_track
        
        # Normalize
        max_val = torch.max(torch.abs(ensemble_signal))
        if max_val > 0:
            ensemble_signal = ensemble_signal / max_val * 0.95
        
        return ensemble_signal

# Example usage and testing
if __name__ == "__main__":
    print("🎵 เริ่มต้นทดสอบเครื่องดนตรีอีสาน")
    
    # Test individual instruments
    sample_rate = 16000
    duration = 2.0
    
    for instrument_name in INSTRUMENT_CHARACTERISTICS.keys():
        print(f"🎼 ทดสอบ {instrument_name}...")
        
        synthesizer = TraditionalInstrumentSynthesizer(instrument_name, sample_rate)
        
        # Test different pitches
        for pitch in [220, 440, 880]:  # A3, A4, A5
            signal = synthesizer(pitch=pitch, duration=duration, velocity=0.7)
            print(f"  🔊 Pitch {pitch}Hz: signal shape {signal.shape}")
    
    # Test ensemble
    print("\n🎭 ทดสอบวงดนตรีอีสาน...")
    
    ensemble_gen = IsanEnsembleGenerator(sample_rate)
    
    # Simple composition
    composition = {
        'พิณ': [
            {'pitch': 440, 'start_time': 0.0, 'duration': 1.0, 'velocity': 0.8},
            {'pitch': 523, 'start_time': 1.0, 'duration': 1.0, 'velocity': 0.7}
        ],
        'แคน': [
            {'pitch': 220, 'start_time': 0.0, 'duration': 2.0, 'velocity': 0.5}
        ],
        'โหวด': [
            {'pitch': 150, 'start_time': 0.0, 'duration': 0.5, 'velocity': 0.9},
            {'pitch': 150, 'start_time': 1.0, 'duration': 0.5, 'velocity': 0.8}
        ],
        'โปงลาง': [
            {'pitch': 330, 'start_time': 0.5, 'duration': 0.8, 'velocity': 0.6}
        ]
    }
    
    ensemble_signal = ensemble_gen(composition, duration=2.0)
    print(f"🎶 สร้างวงดนตรีเสร็จสมบูรณ์: {ensemble_signal.shape}")
    
    print("✅ การทดสอบเสร็จสมบูรณ์")