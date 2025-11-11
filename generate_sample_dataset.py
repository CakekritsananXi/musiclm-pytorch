"""
Generate Sample Isan Music Dataset
สร้างชุดข้อมูลตัวอย่างสำหรับดนตรีอีสาน

This script generates synthetic Isan music samples for testing and demonstration.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse

from isan_instruments import (
    TraditionalInstrumentSynthesizer,
    IsanEnsembleGenerator,
    INSTRUMENT_CHARACTERISTICS
)

class IsanSampleDatasetGenerator:
    """Generate synthetic Isan music samples for training and testing"""
    
    def __init__(self, output_dir: str = "sample_isan_dataset", sample_rate: int = 16000):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        
        # Create directories
        self.audio_dir = self.output_dir / "audio"
        self.metadata_dir = self.output_dir / "metadata"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize synthesizers for each instrument
        self.synthesizers = {}
        for instrument_name, characteristics in INSTRUMENT_CHARACTERISTICS.items():
            self.synthesizers[instrument_name] = TraditionalInstrumentSynthesizer(
                characteristics=characteristics,
                sample_rate=sample_rate
            )
        
        # Initialize ensemble generator
        self.ensemble_generator = IsanEnsembleGenerator(
            instruments=list(INSTRUMENT_CHARACTERISTICS.keys()),
            sample_rate=sample_rate
        )
        
        # Traditional Isan scales (pentatonic)
        self.isan_scales = {
            'major_pentatonic': [0, 2, 4, 7, 9],  # C, D, E, G, A
            'minor_pentatonic': [0, 3, 5, 7, 10],  # C, Eb, F, G, Bb
            'isan_traditional': [0, 2, 5, 7, 9]    # Traditional Isan scale
        }
        
        # Rhythm patterns (in 16th notes)
        self.rhythm_patterns = {
            'basic_4_4': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'syncopated': [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            'lam_rhythm': [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
            'festive': [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
        }
    
    def generate_melody(
        self,
        scale: List[int],
        root_freq: float = 261.63,  # C4
        num_notes: int = 16,
        duration_per_note: float = 0.25
    ) -> List[Dict]:
        """Generate a melodic sequence using a given scale"""
        melody = []
        
        for i in range(num_notes):
            # Choose note from scale
            scale_degree = np.random.choice(scale)
            octave_offset = np.random.choice([0, 12, -12], p=[0.7, 0.2, 0.1])
            midi_offset = scale_degree + octave_offset
            
            # Convert to frequency
            frequency = root_freq * (2 ** (midi_offset / 12))
            
            # Random duration and velocity
            duration = duration_per_note * np.random.uniform(0.8, 1.2)
            velocity = np.random.uniform(0.6, 0.95)
            
            melody.append({
                'frequency': frequency,
                'duration': duration,
                'velocity': velocity
            })
        
        return melody
    
    def generate_single_instrument_sample(
        self,
        instrument: str,
        sample_id: int,
        scale_name: str = 'isan_traditional',
        duration: float = 10.0
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate a sample for a single instrument"""
        
        scale = self.isan_scales[scale_name]
        num_notes = int(duration / 0.25)
        
        # Generate melody
        melody = self.generate_melody(
            scale=scale,
            num_notes=num_notes,
            duration_per_note=0.25
        )
        
        # Synthesize
        synthesizer = self.synthesizers[instrument]
        audio = synthesizer.synthesize_melody(melody)
        
        # Ensure correct duration
        target_samples = int(duration * self.sample_rate)
        if audio.shape[-1] < target_samples:
            padding = target_samples - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        else:
            audio = audio[..., :target_samples]
        
        # Metadata
        metadata = {
            'sample_id': f'{instrument}_{sample_id:04d}',
            'instrument': instrument,
            'scale': scale_name,
            'duration': duration,
            'sample_rate': self.sample_rate,
            'num_notes': len(melody),
            'description': f'Solo {instrument} performance in {scale_name} scale'
        }
        
        return audio, metadata
    
    def generate_ensemble_sample(
        self,
        sample_id: int,
        instruments: List[str],
        duration: float = 15.0,
        style: str = 'traditional'
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate an ensemble performance sample"""
        
        # Generate ensemble performance
        audio = self.ensemble_generator.generate_performance(
            duration=duration,
            style=style,
            tempo=120,
            key='C',
            time_signature=(4, 4)
        )
        
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Metadata
        metadata = {
            'sample_id': f'ensemble_{sample_id:04d}',
            'instruments': instruments,
            'type': 'ensemble',
            'style': style,
            'duration': duration,
            'sample_rate': self.sample_rate,
            'tempo': 120,
            'description': f'Ensemble performance with {", ".join(instruments)}'
        }
        
        return audio, metadata
    
    def save_sample(self, audio: torch.Tensor, metadata: Dict, split: str = 'train'):
        """Save audio and metadata to disk"""
        sample_id = metadata['sample_id']
        
        # Save audio
        audio_path = self.audio_dir / split / f"{sample_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(audio_path), audio, self.sample_rate)
        
        # Save metadata
        metadata_path = self.metadata_dir / split / f"{sample_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def generate_dataset(
        self,
        num_samples_per_instrument: int = 10,
        num_ensemble_samples: int = 20,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ):
        """Generate complete sample dataset"""
        
        print("🎵 Generating Isan Music Sample Dataset...")
        print(f"📁 Output directory: {self.output_dir}")
        
        all_samples = []
        
        # Generate single instrument samples
        print("\n🎸 Generating single instrument samples...")
        for instrument in INSTRUMENT_CHARACTERISTICS.keys():
            print(f"  Generating {num_samples_per_instrument} samples for {instrument}...")
            for i in range(num_samples_per_instrument):
                scale_name = np.random.choice(list(self.isan_scales.keys()))
                duration = np.random.uniform(8.0, 12.0)
                
                audio, metadata = self.generate_single_instrument_sample(
                    instrument=instrument,
                    sample_id=i,
                    scale_name=scale_name,
                    duration=duration
                )
                
                all_samples.append((audio, metadata))
        
        # Generate ensemble samples
        print(f"\n🎼 Generating {num_ensemble_samples} ensemble samples...")
        all_instruments = list(INSTRUMENT_CHARACTERISTICS.keys())
        
        for i in range(num_ensemble_samples):
            # Randomly select 2-4 instruments for ensemble
            num_instruments = np.random.randint(2, 5)
            selected_instruments = np.random.choice(
                all_instruments,
                size=num_instruments,
                replace=False
            ).tolist()
            
            duration = np.random.uniform(12.0, 20.0)
            style = np.random.choice(['traditional', 'festive', 'ceremonial'])
            
            audio, metadata = self.generate_ensemble_sample(
                sample_id=i,
                instruments=selected_instruments,
                duration=duration,
                style=style
            )
            
            all_samples.append((audio, metadata))
        
        # Split dataset
        print(f"\n📊 Splitting dataset (train: {train_val_test_split[0]:.0%}, "
              f"val: {train_val_test_split[1]:.0%}, test: {train_val_test_split[2]:.0%})...")
        
        np.random.shuffle(all_samples)
        n_train = int(len(all_samples) * train_val_test_split[0])
        n_val = int(len(all_samples) * train_val_test_split[1])
        
        train_samples = all_samples[:n_train]
        val_samples = all_samples[n_train:n_train + n_val]
        test_samples = all_samples[n_train + n_val:]
        
        # Save samples
        print("\n💾 Saving samples...")
        for audio, metadata in train_samples:
            self.save_sample(audio, metadata, split='train')
        
        for audio, metadata in val_samples:
            self.save_sample(audio, metadata, split='val')
        
        for audio, metadata in test_samples:
            self.save_sample(audio, metadata, split='test')
        
        # Generate dataset summary
        summary = {
            'total_samples': len(all_samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'instruments': list(INSTRUMENT_CHARACTERISTICS.keys()),
            'sample_rate': self.sample_rate,
            'scales_used': list(self.isan_scales.keys()),
            'generation_date': str(Path(__file__).stat().st_mtime)
        }
        
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   Total samples: {len(all_samples)}")
        print(f"   Train: {len(train_samples)}")
        print(f"   Val: {len(val_samples)}")
        print(f"   Test: {len(test_samples)}")
        print(f"   Summary saved to: {summary_path}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Generate sample Isan music dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='sample_isan_dataset',
        help='Output directory for generated dataset'
    )
    parser.add_argument(
        '--samples-per-instrument',
        type=int,
        default=10,
        help='Number of samples to generate per instrument'
    )
    parser.add_argument(
        '--ensemble-samples',
        type=int,
        default=20,
        help='Number of ensemble samples to generate'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Audio sample rate in Hz'
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = IsanSampleDatasetGenerator(
        output_dir=args.output_dir,
        sample_rate=args.sample_rate
    )
    
    generator.generate_dataset(
        num_samples_per_instrument=args.samples_per_instrument,
        num_ensemble_samples=args.ensemble_samples
    )


if __name__ == '__main__':
    main()
