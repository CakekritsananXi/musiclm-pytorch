# Isan Music Generation - Usage Guide

## Overview

This guide provides comprehensive examples for using the traditional Thai Isan music generation modules with MusicLM.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Models](#training-models)
4. [Generating Music](#generating-music)
5. [Instrument Synthesis](#instrument-synthesis)
6. [Advanced Features](#advanced-features)
7. [Performance Tips](#performance-tips)
8. [Cultural Considerations](#cultural-considerations)

## Quick Start

### Basic Installation

```bash
pip install musiclm-pytorch torch torchaudio librosa pytorch-lightning
```

### Simple Example - Generate Isan Music

```python
import torch
from isan_music_generator import IsanMusicGenerator, IsanMusicConfig
from isan_instruments import IsanEnsembleGenerator

# Create configuration
config = IsanMusicConfig(
    hidden_dim=512,
    num_layers=6,
    sample_rate=16000,
    instruments=['พิณ', 'แคน', 'โหวด', 'โปงลาง']
)

# Initialize generator
generator = IsanMusicGenerator(config)

# Generate music from text description
text_prompt = "Traditional Isan folk song with พิณ leading the melody"
audio = generator.generate_from_text(text_prompt, duration=30.0)

# Save generated audio
import torchaudio
torchaudio.save('isan_music.wav', audio, config.sample_rate)
```

## Dataset Preparation

### Creating a Dataset from Audio Files

```python
from pathlib import Path
from isan_music_dataset import IsanMusicDataset, IsanMusicPreprocessor

# Initialize preprocessor
preprocessor = IsanMusicPreprocessor(
    sample_rate=16000,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

# Process audio files
audio_dir = Path("path/to/isan/music/files")
output_dir = Path("processed_dataset")

preprocessor.process_directory(
    audio_dir=audio_dir,
    output_dir=output_dir,
    file_extensions=['.wav', '.mp3', '.flac']
)

# Create dataset
dataset = IsanMusicDataset(
    data_dir=output_dir,
    transform=None,
    instruments=['พิณ', 'แคน', 'โหวด', 'โปงลาง']
)

print(f"Dataset size: {len(dataset)}")
```

### Using the DataLoader

```python
from isan_music_dataset import create_isan_music_dataloader

# Create dataloader with automatic batching
dataloader = create_isan_music_dataloader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterate through batches
for batch in dataloader:
    audio, metadata = batch
    print(f"Audio shape: {audio.shape}")
    print(f"Instruments: {metadata['instruments']}")
    break
```

## Training Models

### Training with PyTorch Lightning

```python
import pytorch_lightning as pl
from isan_music_generator import IsanMusicGenerator, IsanMusicConfig
from isan_music_dataset import IsanMusicDataset, create_isan_music_dataloader

# Configuration
config = IsanMusicConfig(
    hidden_dim=512,
    num_layers=6,
    learning_rate=1e-4,
    batch_size=16,
    num_epochs=100
)

# Prepare datasets
train_dataset = IsanMusicDataset(data_dir="data/train")
val_dataset = IsanMusicDataset(data_dir="data/val")

train_loader = create_isan_music_dataloader(train_dataset, batch_size=16, shuffle=True)
val_loader = create_isan_music_dataloader(val_dataset, batch_size=16, shuffle=False)

# Initialize model
model = IsanMusicGenerator(config)

# Setup trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    precision=16,
    gradient_clip_val=1.0,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints',
            filename='isan-music-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]
)

# Train model
trainer.fit(model, train_loader, val_loader)
```

### Fine-tuning Pre-trained Model

```python
from isan_music_generator import IsanMusicGenerator

# Load pre-trained checkpoint
model = IsanMusicGenerator.load_from_checkpoint(
    checkpoint_path='checkpoints/isan-music-best.ckpt'
)

# Fine-tune with smaller learning rate
model.config.learning_rate = 1e-5

# Continue training
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model, train_loader, val_loader)
```

## Generating Music

### Text-Conditioned Generation

```python
from isan_music_generator import IsanMusicGenerator

# Load trained model
model = IsanMusicGenerator.load_from_checkpoint('checkpoints/best.ckpt')
model.eval()

# Generate from various text prompts
prompts = [
    "Upbeat Isan folk song with พิณ and แคน",
    "Slow melancholic melody with solo พิณ",
    "Festive ensemble with all instruments playing together",
    "Traditional Lam music with แคน leading"
]

for i, prompt in enumerate(prompts):
    audio = model.generate_from_text(
        text=prompt,
        duration=30.0,
        temperature=1.0,
        top_k=50
    )
    torchaudio.save(f'generated_{i}.wav', audio, model.config.sample_rate)
```

### Instrument-Specific Generation

```python
# Generate music featuring specific instrument
audio = model.generate_with_instrument(
    instrument='พิณ',
    style='traditional',
    duration=20.0,
    tempo=120
)

# Generate with multiple instruments
audio = model.generate_ensemble(
    instruments=['พิณ', 'แคน', 'โปงลาง'],
    interaction_type='call_and_response',
    duration=45.0
)
```

### Controlled Generation Parameters

```python
# Fine-grained control over generation
audio = model.generate(
    text="Traditional Isan melody",
    duration=30.0,
    temperature=0.8,        # Lower = more deterministic
    top_k=40,               # Consider top 40 tokens
    top_p=0.9,              # Nucleus sampling
    repetition_penalty=1.2, # Reduce repetition
    seed=42                 # Reproducible generation
)
```

## Instrument Synthesis

### Individual Instrument Synthesis

```python
from isan_instruments import TraditionalInstrumentSynthesizer, INSTRUMENT_CHARACTERISTICS

# Create synthesizer for พิณ (Phin)
phin_synth = TraditionalInstrumentSynthesizer(
    characteristics=INSTRUMENT_CHARACTERISTICS['พิณ'],
    sample_rate=16000
)

# Generate a note
note = phin_synth.synthesize_note(
    frequency=440.0,  # A4
    duration=1.0,
    velocity=0.8,
    technique='pluck'
)

# Generate a melody
melody_notes = [
    {'frequency': 440.0, 'duration': 0.5, 'velocity': 0.8},
    {'frequency': 493.88, 'duration': 0.5, 'velocity': 0.7},
    {'frequency': 523.25, 'duration': 1.0, 'velocity': 0.9},
]
melody = phin_synth.synthesize_melody(melody_notes)
```

### Ensemble Generation

```python
from isan_instruments import IsanEnsembleGenerator

# Create ensemble generator
ensemble = IsanEnsembleGenerator(
    instruments=['พิณ', 'แคน', 'โหวด', 'โปงลาง'],
    sample_rate=16000
)

# Generate coordinated ensemble performance
performance = ensemble.generate_performance(
    duration=60.0,
    style='traditional_folk',
    tempo=120,
    key='C',
    time_signature=(4, 4)
)

# Control individual instrument volumes
performance = ensemble.generate_with_mix(
    duration=30.0,
    instrument_levels={
        'พิณ': 0.8,   # Lead instrument
        'แคน': 0.6,   # Harmony
        'โหวด': 0.4,  # Rhythm
        'โปงลาง': 0.5 # Percussion
    }
)
```

### Custom Instrument Characteristics

```python
from isan_instruments import InstrumentCharacteristics, TraditionalInstrumentSynthesizer

# Create custom instrument variant
custom_phin = InstrumentCharacteristics(
    name='Modern พิณ',
    frequency_range=(150, 2500),
    harmonic_profile=[1.0, 0.7, 0.4, 0.2, 0.1],
    envelope_params={
        'attack': 0.03,
        'decay': 0.2,
        'sustain': 0.6,
        'release': 1.0
    },
    playing_technique='plucked_string_modern',
    cultural_context='contemporary_fusion'
)

# Use custom characteristics
synth = TraditionalInstrumentSynthesizer(
    characteristics=custom_phin,
    sample_rate=16000
)
```

## Advanced Features

### Rhythm Pattern Generation

```python
from isan_music_generator import IsanMusicGenerator

# Generate with specific rhythm patterns
model = IsanMusicGenerator.load_from_checkpoint('checkpoints/best.ckpt')

# Traditional Isan rhythm patterns
audio = model.generate_with_rhythm(
    rhythm_pattern='lam_ploen',  # Traditional Lam rhythm
    duration=30.0,
    instruments=['พิณ', 'แคน']
)

# Custom rhythm pattern
custom_rhythm = [1, 0, 1, 1, 0, 1, 0, 1]  # Binary pattern
audio = model.generate_with_custom_rhythm(
    rhythm=custom_rhythm,
    tempo=120,
    duration=20.0
)
```

### Cultural Scale and Mode

```python
# Generate using traditional Isan scales
audio = model.generate_with_scale(
    scale='isan_pentatonic',  # Traditional 5-note scale
    root_note='D',
    duration=30.0
)

# Use different musical modes
audio = model.generate_with_mode(
    mode='lam_tang_san',  # Traditional Lam mode
    duration=25.0
)
```

### Multi-Stage Generation

```python
# Generate music in stages for better control
stage1 = model.generate_structure(duration=60.0)  # Generate structure
stage2 = model.add_melody(stage1, instrument='พิณ')  # Add melody
stage3 = model.add_harmony(stage2, instrument='แคน')  # Add harmony
final = model.add_percussion(stage3, instruments=['โหวด', 'โปงลาง'])
```

### Interactive Generation

```python
# Generate music interactively with feedback
from isan_music_generator import InteractiveGenerator

interactive = InteractiveGenerator(model)

# Start with seed audio
seed = model.generate_from_text("Gentle พิณ melody", duration=5.0)

# Continue generation with modifications
continued = interactive.continue_generation(
    seed_audio=seed,
    continuation_duration=15.0,
    modification='add_แคน_harmony'
)

# Interpolate between two styles
style_a = model.generate_from_text("Fast energetic Lam", duration=10.0)
style_b = model.generate_from_text("Slow meditative melody", duration=10.0)

interpolated = interactive.interpolate_styles(
    audio_a=style_a,
    audio_b=style_b,
    num_steps=5
)
```

## Performance Tips

### Batch Processing for Efficiency

```python
# Process multiple prompts efficiently
prompts = [f"Isan song variation {i}" for i in range(10)]

# Batch generation
with torch.no_grad():
    audios = model.batch_generate(
        texts=prompts,
        duration=20.0,
        batch_size=4  # Process 4 at a time
    )
```

### Memory-Efficient Generation

```python
# For long audio generation, use chunking
def generate_long_audio(model, text, total_duration, chunk_duration=30.0):
    chunks = []
    for start in range(0, int(total_duration), int(chunk_duration)):
        chunk = model.generate_from_text(
            text=text,
            duration=min(chunk_duration, total_duration - start),
            seed=start  # Maintain consistency
        )
        chunks.append(chunk)
        torch.cuda.empty_cache()  # Free GPU memory
    
    return torch.cat(chunks, dim=-1)

# Generate 5 minutes of music
long_audio = generate_long_audio(model, "Traditional Isan folk", 300.0)
```

### GPU Optimization

```python
# Enable mixed precision for faster generation
model = model.half()  # Convert to FP16
model = model.cuda()

# Use torch.compile for PyTorch 2.0+
if torch.__version__ >= '2.0.0':
    model = torch.compile(model, mode='reduce-overhead')

# Generate with optimized settings
with torch.cuda.amp.autocast():
    audio = model.generate_from_text("Isan melody", duration=30.0)
```

## Cultural Considerations

### Authentic Instrument Representation

```python
# Ensure authentic sound characteristics
from isan_instruments import validate_instrument_authenticity

# Check if generated audio matches cultural expectations
is_authentic = validate_instrument_authenticity(
    audio=generated_audio,
    instrument='พิณ',
    cultural_criteria={
        'frequency_range_match': True,
        'harmonic_profile_similarity': 0.8,
        'playing_technique_authentic': True
    }
)
```

### Regional Style Variations

```python
# Generate music specific to sub-regions
audio = model.generate_with_regional_style(
    region='central_isan',  # or 'northern_isan', 'southern_isan'
    duration=30.0
)
```

### Traditional Context Awareness

```python
# Generate music appropriate for cultural contexts
contexts = [
    'festival_celebration',
    'religious_ceremony',
    'folk_storytelling',
    'courtship_song',
    'work_song'
]

for context in contexts:
    audio = model.generate_for_context(
        context=context,
        duration=30.0
    )
    torchaudio.save(f'{context}.wav', audio, model.config.sample_rate)
```

## Complete Example: End-to-End Pipeline

```python
import torch
import torchaudio
import pytorch_lightning as pl
from pathlib import Path

from isan_music_dataset import IsanMusicDataset, IsanMusicPreprocessor, create_isan_music_dataloader
from isan_music_generator import IsanMusicGenerator, IsanMusicConfig
from isan_instruments import IsanEnsembleGenerator

# 1. Preprocess raw audio files
preprocessor = IsanMusicPreprocessor()
preprocessor.process_directory(
    audio_dir=Path("raw_audio"),
    output_dir=Path("processed")
)

# 2. Create datasets
train_dataset = IsanMusicDataset(data_dir="processed/train")
val_dataset = IsanMusicDataset(data_dir="processed/val")

train_loader = create_isan_music_dataloader(train_dataset, batch_size=16)
val_loader = create_isan_music_dataloader(val_dataset, batch_size=16)

# 3. Configure and train model
config = IsanMusicConfig(
    hidden_dim=512,
    num_layers=6,
    learning_rate=1e-4,
    instruments=['พิณ', 'แคน', 'โหวด', 'โปงลาง']
)

model = IsanMusicGenerator(config)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    precision=16,
    callbacks=[pl.callbacks.ModelCheckpoint(monitor='val_loss')]
)

trainer.fit(model, train_loader, val_loader)

# 4. Generate music
model.eval()
audio = model.generate_from_text(
    "Traditional Isan ensemble performance",
    duration=60.0
)

# 5. Save result
torchaudio.save('isan_generation.wav', audio, config.sample_rate)

print("Generation complete!")
```

## Troubleshooting

### Common Issues

**Issue: Low quality audio**
```python
# Solution: Increase model capacity
config.hidden_dim = 1024
config.num_layers = 12
```

**Issue: Instrument doesn't sound authentic**
```python
# Solution: Adjust instrument characteristics
from isan_instruments import INSTRUMENT_CHARACTERISTICS
chars = INSTRUMENT_CHARACTERISTICS['พิณ']
chars.harmonic_profile = [1.0, 0.65, 0.35, 0.18, 0.09]  # Tweak harmonics
```

**Issue: Out of memory during training**
```python
# Solution: Reduce batch size and use gradient accumulation
config.batch_size = 4
trainer = pl.Trainer(accumulate_grad_batches=4)  # Effective batch size = 16
```

## Additional Resources

- **Cultural Reference**: Consult with Thai musicologists for authentic representation
- **Dataset Sources**: Traditional Isan music recordings from Thai cultural archives
- **Community**: Join discussions on traditional Thai music preservation

## Citation

If you use these modules in your research, please cite:

```bibtex
@software{isan_musiclm_2024,
    title = {Traditional Thai Isan Music Generation with MusicLM},
    author = {MusicLM-PyTorch Contributors},
    year = {2024},
    url = {https://github.com/CakekritsananXi/musiclm-pytorch}
}
```

## License

This implementation respects and honors traditional Thai Isan musical heritage. Please use responsibly and with cultural sensitivity.
