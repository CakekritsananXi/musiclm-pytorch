"""
Cultural Validation Framework for Isan Music
กรอบการตรวจสอบความถูกต้องทางวัฒนธรรมสำหรับดนตรีอีสาน

This module provides tools to validate that generated music respects and
accurately represents traditional Isan musical culture.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

from isan_instruments import INSTRUMENT_CHARACTERISTICS


@dataclass
class CulturalCriteria:
    """Criteria for cultural validation"""
    
    # Instrument authenticity
    frequency_range_tolerance: float = 0.15  # 15% tolerance
    harmonic_profile_similarity_threshold: float = 0.75
    
    # Musical structure
    scale_adherence_threshold: float = 0.80
    rhythm_pattern_accuracy_threshold: float = 0.70
    
    # Performance characteristics
    tempo_range: Tuple[int, int] = (80, 160)  # BPM
    dynamics_range: Tuple[float, float] = (0.3, 0.95)
    
    # Ensemble characteristics
    instrument_balance_tolerance: float = 0.20
    acceptable_instrument_combinations: List[List[str]] = field(default_factory=list)
    
    # Regional style markers
    regional_characteristics: Dict[str, Any] = field(default_factory=dict)


# Default cultural criteria for Isan music
DEFAULT_ISAN_CRITERIA = CulturalCriteria(
    acceptable_instrument_combinations=[
        ['พิณ'],
        ['แคน'],
        ['พิณ', 'แคน'],
        ['พิณ', 'แคน', 'โหวด'],
        ['พิณ', 'แคน', 'โปงลาง'],
        ['พิณ', 'แคน', 'โหวด', 'โปงลาง'],
        ['แคน', 'โหวด', 'โปงลาง'],
    ],
    regional_characteristics={
        'central_isan': {
            'preferred_scales': ['isan_traditional', 'major_pentatonic'],
            'tempo_range': (100, 140),
            'lead_instruments': ['พิณ', 'แคน']
        },
        'northern_isan': {
            'preferred_scales': ['minor_pentatonic'],
            'tempo_range': (90, 130),
            'lead_instruments': ['แคน']
        },
        'southern_isan': {
            'preferred_scales': ['isan_traditional'],
            'tempo_range': (110, 150),
            'lead_instruments': ['พิณ']
        }
    }
)


class InstrumentAuthenticityValidator:
    """Validate instrument sound authenticity"""
    
    def __init__(self, instrument_name: str):
        """
        Initialize validator for specific instrument
        
        Args:
            instrument_name: Name of the instrument to validate
        """
        self.instrument_name = instrument_name
        self.characteristics = INSTRUMENT_CHARACTERISTICS.get(instrument_name)
        
        if self.characteristics is None:
            raise ValueError(f"Unknown instrument: {instrument_name}")
    
    def validate_frequency_range(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        tolerance: float = 0.15
    ) -> Dict[str, Any]:
        """
        Validate that audio frequencies are within expected range
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate in Hz
            tolerance: Tolerance for range (e.g., 0.15 = 15%)
            
        Returns:
            Validation results
        """
        # Perform FFT
        audio_np = audio.squeeze().numpy()
        fft = np.fft.rfft(audio_np)
        freqs = np.fft.rfftfreq(len(audio_np), 1/sample_rate)
        magnitudes = np.abs(fft)
        
        # Find significant frequencies (above threshold)
        threshold = np.max(magnitudes) * 0.1
        significant_freqs = freqs[magnitudes > threshold]
        
        # Expected frequency range
        min_freq, max_freq = self.characteristics.frequency_range
        min_freq_tolerant = min_freq * (1 - tolerance)
        max_freq_tolerant = max_freq * (1 + tolerance)
        
        # Check if significant frequencies are within range
        in_range = np.logical_and(
            significant_freqs >= min_freq_tolerant,
            significant_freqs <= max_freq_tolerant
        )
        
        compliance_rate = np.mean(in_range) if len(in_range) > 0 else 0.0
        
        return {
            'instrument': self.instrument_name,
            'is_valid': compliance_rate >= 0.8,
            'compliance_rate': float(compliance_rate),
            'expected_range': (min_freq, max_freq),
            'actual_range': (float(np.min(significant_freqs)), float(np.max(significant_freqs))),
            'num_significant_frequencies': len(significant_freqs),
            'details': 'Frequency range within acceptable bounds' if compliance_rate >= 0.8 
                      else 'Frequency range deviates from expected'
        }
    
    def validate_harmonic_profile(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        fundamental_freq: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate harmonic profile matches expected characteristics
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate in Hz
            fundamental_freq: Expected fundamental frequency (optional)
            
        Returns:
            Validation results
        """
        # Perform FFT
        audio_np = audio.squeeze().numpy()
        fft = np.fft.rfft(audio_np)
        freqs = np.fft.rfftfreq(len(audio_np), 1/sample_rate)
        magnitudes = np.abs(fft)
        
        # Find fundamental frequency if not provided
        if fundamental_freq is None:
            fundamental_idx = np.argmax(magnitudes)
            fundamental_freq = freqs[fundamental_idx]
        
        # Extract harmonics
        expected_harmonics = self.characteristics.harmonic_profile
        actual_harmonics = []
        
        for i in range(1, len(expected_harmonics) + 1):
            harmonic_freq = fundamental_freq * i
            # Find closest frequency bin
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            if idx < len(magnitudes):
                actual_harmonics.append(magnitudes[idx])
            else:
                actual_harmonics.append(0)
        
        # Normalize
        if len(actual_harmonics) > 0 and actual_harmonics[0] > 0:
            actual_harmonics = np.array(actual_harmonics) / actual_harmonics[0]
            expected_harmonics_np = np.array(expected_harmonics[:len(actual_harmonics)])
            
            # Calculate similarity (cosine similarity)
            similarity = np.dot(actual_harmonics, expected_harmonics_np) / (
                np.linalg.norm(actual_harmonics) * np.linalg.norm(expected_harmonics_np) + 1e-8
            )
        else:
            similarity = 0.0
        
        return {
            'instrument': self.instrument_name,
            'is_valid': similarity >= 0.75,
            'similarity_score': float(similarity),
            'fundamental_frequency': float(fundamental_freq),
            'expected_harmonics': expected_harmonics,
            'actual_harmonics': actual_harmonics.tolist() if isinstance(actual_harmonics, np.ndarray) else actual_harmonics,
            'details': 'Harmonic profile matches expected' if similarity >= 0.75 
                      else 'Harmonic profile differs from expected'
        }
    
    def validate_envelope(
        self,
        audio: torch.Tensor,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Validate ADSR envelope characteristics
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate in Hz
            
        Returns:
            Validation results
        """
        audio_np = audio.squeeze().numpy()
        
        # Calculate envelope (amplitude over time)
        window_size = int(0.01 * sample_rate)  # 10ms windows
        envelope = []
        
        for i in range(0, len(audio_np), window_size):
            window = audio_np[i:i+window_size]
            if len(window) > 0:
                rms = np.sqrt(np.mean(window ** 2))
                envelope.append(rms)
        
        envelope = np.array(envelope)
        
        if len(envelope) < 4:
            return {
                'instrument': self.instrument_name,
                'is_valid': False,
                'details': 'Audio too short to validate envelope'
            }
        
        # Find attack peak
        attack_peak_idx = np.argmax(envelope[:len(envelope)//3])
        attack_time = attack_peak_idx * window_size / sample_rate
        
        # Expected envelope parameters
        expected_attack = self.characteristics.envelope_params['attack']
        attack_tolerance = 0.5  # seconds
        
        attack_valid = abs(attack_time - expected_attack) <= attack_tolerance
        
        return {
            'instrument': self.instrument_name,
            'is_valid': attack_valid,
            'attack_time': float(attack_time),
            'expected_attack': expected_attack,
            'envelope_shape': envelope.tolist()[:20],  # First 20 values
            'details': 'Envelope characteristics match' if attack_valid 
                      else 'Envelope differs from expected'
        }


class MusicalStructureValidator:
    """Validate musical structure and patterns"""
    
    @staticmethod
    def validate_scale_adherence(
        note_frequencies: List[float],
        scale_notes: List[float],
        tolerance_cents: float = 50
    ) -> Dict[str, Any]:
        """
        Validate that notes adhere to a given scale
        
        Args:
            note_frequencies: List of detected note frequencies
            scale_notes: Expected scale note frequencies
            tolerance_cents: Tolerance in cents (100 cents = 1 semitone)
            
        Returns:
            Validation results
        """
        if len(note_frequencies) == 0:
            return {
                'is_valid': False,
                'details': 'No notes detected'
            }
        
        # Check each note against scale
        adherent_notes = 0
        
        for freq in note_frequencies:
            # Find closest scale note
            cents_diff = [
                1200 * np.log2(freq / scale_note)
                for scale_note in scale_notes
            ]
            min_diff = min(abs(d) for d in cents_diff)
            
            if min_diff <= tolerance_cents:
                adherent_notes += 1
        
        adherence_rate = adherent_notes / len(note_frequencies)
        
        return {
            'is_valid': adherence_rate >= 0.80,
            'adherence_rate': adherence_rate,
            'total_notes': len(note_frequencies),
            'adherent_notes': adherent_notes,
            'tolerance_cents': tolerance_cents,
            'details': f'{adherence_rate*100:.1f}% of notes adhere to scale'
        }
    
    @staticmethod
    def validate_rhythm_pattern(
        audio: torch.Tensor,
        sample_rate: int,
        expected_pattern: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Validate rhythm pattern
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate in Hz
            expected_pattern: Expected rhythm pattern (list of 0s and 1s)
            
        Returns:
            Validation results
        """
        audio_np = audio.squeeze().numpy()
        
        # Detect onsets using energy-based method
        window_size = int(0.05 * sample_rate)  # 50ms windows
        energies = []
        
        for i in range(0, len(audio_np), window_size):
            window = audio_np[i:i+window_size]
            if len(window) > 0:
                energy = np.sum(window ** 2)
                energies.append(energy)
        
        energies = np.array(energies)
        
        # Detect onsets (local maxima above threshold)
        threshold = np.mean(energies) + 0.5 * np.std(energies)
        onsets = energies > threshold
        
        # Calculate rhythm metrics
        onset_count = np.sum(onsets)
        onset_regularity = np.std(np.diff(np.where(onsets)[0])) if onset_count > 1 else 0
        
        return {
            'is_valid': onset_count > 0,
            'onset_count': int(onset_count),
            'rhythm_regularity': float(onset_regularity),
            'has_rhythm': onset_count > 0,
            'details': f'Detected {onset_count} rhythmic onsets'
        }
    
    @staticmethod
    def validate_tempo(
        audio: torch.Tensor,
        sample_rate: int,
        expected_range: Tuple[int, int] = (80, 160)
    ) -> Dict[str, Any]:
        """
        Validate tempo is within expected range
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate in Hz
            expected_range: Expected BPM range (min, max)
            
        Returns:
            Validation results
        """
        audio_np = audio.squeeze().numpy()
        
        # Simple onset detection
        window_size = int(0.05 * sample_rate)
        energies = []
        
        for i in range(0, len(audio_np), window_size):
            window = audio_np[i:i+window_size]
            if len(window) > 0:
                energies.append(np.sum(window ** 2))
        
        energies = np.array(energies)
        threshold = np.mean(energies) + 0.5 * np.std(energies)
        onsets = np.where(energies > threshold)[0]
        
        if len(onsets) < 2:
            return {
                'is_valid': False,
                'estimated_tempo': 0,
                'details': 'Not enough onsets to estimate tempo'
            }
        
        # Calculate inter-onset intervals
        intervals = np.diff(onsets) * window_size / sample_rate  # in seconds
        avg_interval = np.median(intervals)
        
        # Convert to BPM
        estimated_bpm = 60.0 / avg_interval if avg_interval > 0 else 0
        
        # Validate
        is_valid = expected_range[0] <= estimated_bpm <= expected_range[1]
        
        return {
            'is_valid': is_valid,
            'estimated_tempo': float(estimated_bpm),
            'expected_range': expected_range,
            'details': f'Tempo {"within" if is_valid else "outside"} expected range'
        }


class EnsembleValidator:
    """Validate ensemble performances"""
    
    @staticmethod
    def validate_instrument_combination(
        instruments: List[str],
        criteria: CulturalCriteria = DEFAULT_ISAN_CRITERIA
    ) -> Dict[str, Any]:
        """
        Validate that instrument combination is culturally appropriate
        
        Args:
            instruments: List of instrument names
            criteria: Cultural criteria
            
        Returns:
            Validation results
        """
        instruments_sorted = sorted(instruments)
        
        # Check if combination is acceptable
        is_acceptable = any(
            sorted(combo) == instruments_sorted
            for combo in criteria.acceptable_instrument_combinations
        )
        
        return {
            'is_valid': is_acceptable,
            'instruments': instruments,
            'details': 'Instrument combination is culturally authentic' if is_acceptable
                      else 'Unusual instrument combination - may not be traditional'
        }
    
    @staticmethod
    def validate_instrument_balance(
        audio_tracks: Dict[str, torch.Tensor],
        tolerance: float = 0.20
    ) -> Dict[str, Any]:
        """
        Validate balance between instruments in ensemble
        
        Args:
            audio_tracks: Dictionary mapping instrument names to audio tracks
            tolerance: Acceptable level difference tolerance
            
        Returns:
            Validation results
        """
        # Calculate RMS level for each track
        levels = {}
        for instrument, audio in audio_tracks.items():
            rms = torch.sqrt(torch.mean(audio ** 2))
            levels[instrument] = float(rms)
        
        if len(levels) < 2:
            return {
                'is_valid': True,
                'details': 'Single instrument, no balance to check'
            }
        
        # Check if levels are within tolerance of each other
        max_level = max(levels.values())
        min_level = min(levels.values())
        
        level_difference = (max_level - min_level) / (max_level + 1e-8)
        is_balanced = level_difference <= tolerance
        
        return {
            'is_valid': is_balanced,
            'instrument_levels': levels,
            'level_difference': float(level_difference),
            'tolerance': tolerance,
            'details': 'Instruments well balanced' if is_balanced
                      else 'Some instruments may be too loud or quiet'
        }


class CulturalValidationReport:
    """Generate comprehensive validation report"""
    
    def __init__(self, criteria: CulturalCriteria = DEFAULT_ISAN_CRITERIA):
        """
        Initialize validation report
        
        Args:
            criteria: Cultural validation criteria
        """
        self.criteria = criteria
        self.results = []
    
    def add_validation(self, category: str, result: Dict[str, Any]):
        """Add validation result"""
        self.results.append({
            'category': category,
            'result': result,
            'is_valid': result.get('is_valid', False)
        })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        total_validations = len(self.results)
        passed_validations = sum(1 for r in self.results if r['is_valid'])
        
        overall_score = passed_validations / total_validations if total_validations > 0 else 0
        
        report = {
            'overall_valid': overall_score >= 0.75,
            'overall_score': overall_score,
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'failed_validations': total_validations - passed_validations,
            'validations': self.results,
            'summary': self._generate_summary(overall_score)
        }
        
        return report
    
    def _generate_summary(self, score: float) -> str:
        """Generate human-readable summary"""
        if score >= 0.90:
            return "Excellent cultural authenticity - highly representative of traditional Isan music"
        elif score >= 0.75:
            return "Good cultural authenticity - generally representative with minor deviations"
        elif score >= 0.60:
            return "Moderate cultural authenticity - some significant deviations from tradition"
        else:
            return "Low cultural authenticity - substantial deviations from traditional Isan music"
    
    def save_report(self, filepath: str):
        """Save report to JSON file"""
        report = self.generate_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    def print_report(self):
        """Print formatted report to console"""
        report = self.generate_report()
        
        print("\n" + "="*70)
        print("ISAN MUSIC CULTURAL VALIDATION REPORT")
        print("="*70)
        print(f"\nOverall Score: {report['overall_score']*100:.1f}%")
        print(f"Status: {'✅ VALID' if report['overall_valid'] else '⚠️  NEEDS REVIEW'}")
        print(f"\nPassed: {report['passed_validations']}/{report['total_validations']}")
        print(f"\nSummary: {report['summary']}")
        print("\n" + "-"*70)
        print("Detailed Results:")
        print("-"*70)
        
        for validation in report['validations']:
            status = "✅" if validation['is_valid'] else "❌"
            print(f"\n{status} {validation['category']}")
            print(f"   {validation['result'].get('details', 'No details')}")
        
        print("\n" + "="*70)


if __name__ == '__main__':
    print("Isan Music Cultural Validation Framework")
    print("กรอบการตรวจสอบความถูกต้องทางวัฒนธรรมสำหรับดนตรีอีสาน")
    print("="*70)
    print("\nThis framework helps ensure generated music respects")
    print("traditional Isan musical culture and characteristics.")
