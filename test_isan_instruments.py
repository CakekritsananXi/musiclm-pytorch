"""
Unit Tests for Isan Instrument Models
การทดสอบโมเดลเครื่องดนตรีอีสาน
"""

import unittest
import torch
import numpy as np
from typing import Dict

from isan_instruments import (
    TraditionalInstrumentSynthesizer,
    IsanEnsembleGenerator,
    INSTRUMENT_CHARACTERISTICS,
    InstrumentCharacteristics
)


class TestInstrumentCharacteristics(unittest.TestCase):
    """Test InstrumentCharacteristics dataclass"""
    
    def test_all_instruments_defined(self):
        """Test that all expected instruments are defined"""
        expected_instruments = ['พิณ', 'แคน', 'โหวด', 'โปงลาง']
        for instrument in expected_instruments:
            self.assertIn(instrument, INSTRUMENT_CHARACTERISTICS)
    
    def test_characteristics_validity(self):
        """Test that all characteristics have valid parameters"""
        for name, chars in INSTRUMENT_CHARACTERISTICS.items():
            # Check frequency range
            self.assertIsInstance(chars.frequency_range, tuple)
            self.assertEqual(len(chars.frequency_range), 2)
            self.assertGreater(chars.frequency_range[1], chars.frequency_range[0])
            self.assertGreater(chars.frequency_range[0], 0)
            
            # Check harmonic profile
            self.assertIsInstance(chars.harmonic_profile, list)
            self.assertGreater(len(chars.harmonic_profile), 0)
            self.assertEqual(chars.harmonic_profile[0], 1.0)  # Fundamental should be 1.0
            
            # Check envelope parameters
            self.assertIsInstance(chars.envelope_params, dict)
            required_params = ['attack', 'decay', 'sustain', 'release']
            for param in required_params:
                self.assertIn(param, chars.envelope_params)
                self.assertGreater(chars.envelope_params[param], 0)
    
    def test_phin_characteristics(self):
        """Test specific characteristics of พิณ (Phin)"""
        phin = INSTRUMENT_CHARACTERISTICS['พิณ']
        self.assertEqual(phin.name, 'พิณ')
        self.assertEqual(phin.playing_technique, 'plucked_string')
        self.assertGreater(phin.frequency_range[0], 150)  # At least 150 Hz
        self.assertLess(phin.frequency_range[1], 2500)   # Less than 2500 Hz
    
    def test_khaen_characteristics(self):
        """Test specific characteristics of แคน (Khaen)"""
        khaen = INSTRUMENT_CHARACTERISTICS['แคน']
        self.assertEqual(khaen.name, 'แคน')
        self.assertIn('reed', khaen.playing_technique.lower())
        # Khaen has lower frequency range
        self.assertLess(khaen.frequency_range[0], 100)


class TestTraditionalInstrumentSynthesizer(unittest.TestCase):
    """Test TraditionalInstrumentSynthesizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.phin_chars = INSTRUMENT_CHARACTERISTICS['พิณ']
        self.synth = TraditionalInstrumentSynthesizer(
            characteristics=self.phin_chars,
            sample_rate=self.sample_rate
        )
    
    def test_initialization(self):
        """Test synthesizer initialization"""
        self.assertEqual(self.synth.sample_rate, self.sample_rate)
        self.assertEqual(self.synth.characteristics, self.phin_chars)
    
    def test_synthesize_note_output_shape(self):
        """Test that synthesize_note produces correct output shape"""
        frequency = 440.0  # A4
        duration = 1.0
        
        audio = self.synth.synthesize_note(
            frequency=frequency,
            duration=duration,
            velocity=0.8
        )
        
        # Check output is tensor
        self.assertIsInstance(audio, torch.Tensor)
        
        # Check shape - should be (1, num_samples)
        expected_samples = int(duration * self.sample_rate)
        self.assertEqual(audio.dim(), 2)
        self.assertEqual(audio.shape[0], 1)
        self.assertAlmostEqual(audio.shape[1], expected_samples, delta=1000)
    
    def test_synthesize_note_frequency_range(self):
        """Test synthesizing notes at different frequencies"""
        duration = 0.5
        
        # Test low frequency
        audio_low = self.synth.synthesize_note(
            frequency=self.phin_chars.frequency_range[0],
            duration=duration
        )
        self.assertIsNotNone(audio_low)
        
        # Test high frequency
        audio_high = self.synth.synthesize_note(
            frequency=self.phin_chars.frequency_range[1],
            duration=duration
        )
        self.assertIsNotNone(audio_high)
        
        # Test mid frequency
        mid_freq = (self.phin_chars.frequency_range[0] + 
                   self.phin_chars.frequency_range[1]) / 2
        audio_mid = self.synth.synthesize_note(
            frequency=mid_freq,
            duration=duration
        )
        self.assertIsNotNone(audio_mid)
    
    def test_synthesize_note_velocity_effect(self):
        """Test that velocity affects amplitude"""
        frequency = 440.0
        duration = 0.5
        
        audio_soft = self.synth.synthesize_note(frequency, duration, velocity=0.3)
        audio_loud = self.synth.synthesize_note(frequency, duration, velocity=0.9)
        
        # Louder velocity should produce higher RMS amplitude
        rms_soft = torch.sqrt(torch.mean(audio_soft ** 2))
        rms_loud = torch.sqrt(torch.mean(audio_loud ** 2))
        
        self.assertGreater(rms_loud, rms_soft)
    
    def test_synthesize_melody(self):
        """Test melody synthesis"""
        melody = [
            {'frequency': 440.0, 'duration': 0.5, 'velocity': 0.8},
            {'frequency': 493.88, 'duration': 0.5, 'velocity': 0.7},
            {'frequency': 523.25, 'duration': 1.0, 'velocity': 0.9},
        ]
        
        audio = self.synth.synthesize_melody(melody)
        
        # Check output
        self.assertIsInstance(audio, torch.Tensor)
        expected_duration = sum(note['duration'] for note in melody)
        expected_samples = int(expected_duration * self.sample_rate)
        self.assertAlmostEqual(audio.shape[-1], expected_samples, delta=2000)
    
    def test_empty_melody(self):
        """Test handling of empty melody"""
        audio = self.synth.synthesize_melody([])
        self.assertIsInstance(audio, torch.Tensor)
        self.assertGreater(audio.shape[-1], 0)  # Should return silence, not crash
    
    def test_different_instruments(self):
        """Test synthesizers for all instruments"""
        for instrument_name, characteristics in INSTRUMENT_CHARACTERISTICS.items():
            synth = TraditionalInstrumentSynthesizer(
                characteristics=characteristics,
                sample_rate=self.sample_rate
            )
            
            # Generate a note
            audio = synth.synthesize_note(
                frequency=440.0,
                duration=0.5,
                velocity=0.8
            )
            
            # Verify output
            self.assertIsInstance(audio, torch.Tensor)
            self.assertGreater(audio.shape[-1], 0)
            self.assertFalse(torch.isnan(audio).any())
            self.assertFalse(torch.isinf(audio).any())


class TestIsanEnsembleGenerator(unittest.TestCase):
    """Test IsanEnsembleGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.instruments = ['พิณ', 'แคน', 'โหวด', 'โปงลาง']
        self.ensemble = IsanEnsembleGenerator(
            instruments=self.instruments,
            sample_rate=self.sample_rate
        )
    
    def test_initialization(self):
        """Test ensemble generator initialization"""
        self.assertEqual(self.ensemble.sample_rate, self.sample_rate)
        self.assertEqual(len(self.ensemble.synthesizers), len(self.instruments))
    
    def test_generate_performance(self):
        """Test ensemble performance generation"""
        duration = 5.0
        
        audio = self.ensemble.generate_performance(
            duration=duration,
            style='traditional',
            tempo=120,
            key='C',
            time_signature=(4, 4)
        )
        
        # Check output
        self.assertIsInstance(audio, torch.Tensor)
        expected_samples = int(duration * self.sample_rate)
        self.assertAlmostEqual(audio.shape[-1], expected_samples, delta=5000)
        
        # Check audio is valid
        self.assertFalse(torch.isnan(audio).any())
        self.assertFalse(torch.isinf(audio).any())
    
    def test_generate_with_mix(self):
        """Test ensemble generation with custom mix levels"""
        duration = 3.0
        instrument_levels = {
            'พิณ': 0.8,
            'แคน': 0.6,
            'โหวด': 0.4,
            'โปงลาง': 0.5
        }
        
        audio = self.ensemble.generate_with_mix(
            duration=duration,
            instrument_levels=instrument_levels
        )
        
        self.assertIsInstance(audio, torch.Tensor)
        self.assertFalse(torch.isnan(audio).any())
    
    def test_different_styles(self):
        """Test generation with different styles"""
        styles = ['traditional', 'festive', 'ceremonial']
        duration = 2.0
        
        for style in styles:
            audio = self.ensemble.generate_performance(
                duration=duration,
                style=style,
                tempo=120
            )
            
            self.assertIsInstance(audio, torch.Tensor)
            self.assertGreater(audio.shape[-1], 0)
    
    def test_different_tempos(self):
        """Test generation with different tempos"""
        tempos = [80, 120, 160]
        duration = 2.0
        
        for tempo in tempos:
            audio = self.ensemble.generate_performance(
                duration=duration,
                style='traditional',
                tempo=tempo
            )
            
            self.assertIsInstance(audio, torch.Tensor)
            self.assertGreater(audio.shape[-1], 0)
    
    def test_subset_of_instruments(self):
        """Test ensemble with subset of instruments"""
        subset = ['พิณ', 'แคน']
        ensemble = IsanEnsembleGenerator(
            instruments=subset,
            sample_rate=self.sample_rate
        )
        
        audio = ensemble.generate_performance(
            duration=3.0,
            style='traditional',
            tempo=120
        )
        
        self.assertIsInstance(audio, torch.Tensor)
        self.assertGreater(audio.shape[-1], 0)
    
    def test_single_instrument_ensemble(self):
        """Test ensemble with single instrument"""
        ensemble = IsanEnsembleGenerator(
            instruments=['พิณ'],
            sample_rate=self.sample_rate
        )
        
        audio = ensemble.generate_performance(
            duration=2.0,
            style='traditional',
            tempo=120
        )
        
        self.assertIsInstance(audio, torch.Tensor)


class TestAudioQuality(unittest.TestCase):
    """Test audio quality metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.synth = TraditionalInstrumentSynthesizer(
            characteristics=INSTRUMENT_CHARACTERISTICS['พิณ'],
            sample_rate=self.sample_rate
        )
    
    def test_no_clipping(self):
        """Test that audio doesn't clip"""
        audio = self.synth.synthesize_note(440.0, 1.0, velocity=0.9)
        
        # Check for clipping (values > 1.0 or < -1.0)
        self.assertLessEqual(audio.abs().max().item(), 1.0)
    
    def test_no_silence_gaps(self):
        """Test melody has no unexpected silence gaps"""
        melody = [
            {'frequency': 440.0, 'duration': 0.5, 'velocity': 0.8},
            {'frequency': 493.88, 'duration': 0.5, 'velocity': 0.8},
        ]
        
        audio = self.synth.synthesize_melody(melody)
        
        # Calculate RMS over time windows
        window_size = 1000
        rms_values = []
        for i in range(0, audio.shape[-1] - window_size, window_size):
            window = audio[..., i:i+window_size]
            rms = torch.sqrt(torch.mean(window ** 2))
            rms_values.append(rms.item())
        
        # Should not have many consecutive very quiet windows
        quiet_threshold = 0.01
        quiet_windows = [rms < quiet_threshold for rms in rms_values]
        max_consecutive_quiet = max(
            (sum(1 for _ in group) for key, group in 
             __import__('itertools').groupby(quiet_windows) if key),
            default=0
        )
        
        # Allow some quiet windows but not too many consecutive
        self.assertLess(max_consecutive_quiet, 5)
    
    def test_frequency_content(self):
        """Test that generated audio has expected frequency content"""
        target_freq = 440.0
        duration = 1.0
        
        audio = self.synth.synthesize_note(target_freq, duration, velocity=0.8)
        
        # Perform FFT
        audio_np = audio.squeeze().numpy()
        fft = np.fft.rfft(audio_np)
        freqs = np.fft.rfftfreq(len(audio_np), 1/self.sample_rate)
        magnitudes = np.abs(fft)
        
        # Find peak frequency
        peak_idx = np.argmax(magnitudes)
        peak_freq = freqs[peak_idx]
        
        # Peak should be close to target frequency
        self.assertAlmostEqual(peak_freq, target_freq, delta=20)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.synth = TraditionalInstrumentSynthesizer(
            characteristics=INSTRUMENT_CHARACTERISTICS['พิณ'],
            sample_rate=self.sample_rate
        )
    
    def test_zero_duration(self):
        """Test handling of zero duration"""
        audio = self.synth.synthesize_note(440.0, 0.0, velocity=0.8)
        self.assertIsInstance(audio, torch.Tensor)
    
    def test_very_short_duration(self):
        """Test very short duration"""
        audio = self.synth.synthesize_note(440.0, 0.001, velocity=0.8)
        self.assertIsInstance(audio, torch.Tensor)
        self.assertGreater(audio.shape[-1], 0)
    
    def test_very_long_duration(self):
        """Test very long duration"""
        audio = self.synth.synthesize_note(440.0, 60.0, velocity=0.8)
        self.assertIsInstance(audio, torch.Tensor)
        expected_samples = int(60.0 * self.sample_rate)
        self.assertAlmostEqual(audio.shape[-1], expected_samples, delta=5000)
    
    def test_extreme_frequencies(self):
        """Test extreme but valid frequencies"""
        # Very low frequency (but audible)
        audio_low = self.synth.synthesize_note(20.0, 0.5, velocity=0.8)
        self.assertIsInstance(audio_low, torch.Tensor)
        
        # Very high frequency
        audio_high = self.synth.synthesize_note(5000.0, 0.5, velocity=0.8)
        self.assertIsInstance(audio_high, torch.Tensor)
    
    def test_zero_velocity(self):
        """Test zero velocity"""
        audio = self.synth.synthesize_note(440.0, 0.5, velocity=0.0)
        self.assertIsInstance(audio, torch.Tensor)
        # Should produce very quiet or silent audio
        rms = torch.sqrt(torch.mean(audio ** 2))
        self.assertLess(rms.item(), 0.1)
    
    def test_max_velocity(self):
        """Test maximum velocity"""
        audio = self.synth.synthesize_note(440.0, 0.5, velocity=1.0)
        self.assertIsInstance(audio, torch.Tensor)
        # Should not clip
        self.assertLessEqual(audio.abs().max().item(), 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components"""
    
    def test_full_pipeline(self):
        """Test complete pipeline from synthesis to ensemble"""
        sample_rate = 16000
        
        # Create individual synthesizers
        synthesizers = {}
        for name, chars in INSTRUMENT_CHARACTERISTICS.items():
            synthesizers[name] = TraditionalInstrumentSynthesizer(
                characteristics=chars,
                sample_rate=sample_rate
            )
        
        # Generate notes for each instrument
        for name, synth in synthesizers.items():
            audio = synth.synthesize_note(440.0, 1.0, velocity=0.8)
            self.assertIsInstance(audio, torch.Tensor)
        
        # Create ensemble
        ensemble = IsanEnsembleGenerator(
            instruments=list(INSTRUMENT_CHARACTERISTICS.keys()),
            sample_rate=sample_rate
        )
        
        # Generate ensemble performance
        performance = ensemble.generate_performance(
            duration=5.0,
            style='traditional',
            tempo=120
        )
        
        self.assertIsInstance(performance, torch.Tensor)
        self.assertFalse(torch.isnan(performance).any())


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInstrumentCharacteristics))
    suite.addTests(loader.loadTestsFromTestCase(TestTraditionalInstrumentSynthesizer))
    suite.addTests(loader.loadTestsFromTestCase(TestIsanEnsembleGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result


if __name__ == '__main__':
    run_tests()
