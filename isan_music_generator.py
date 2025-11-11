"""
Isan Music Generation and Training Module
โมเดลสำหรับสร้างและฝึกสอน AI สำหรับดนตรีอีสาน
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime

from isan_music_dataset import IsanMusicDataset, IsanMusicPreprocessor, create_isan_music_dataloader
from isan_instruments import (
    TraditionalInstrumentSynthesizer, 
    IsanEnsembleGenerator,
    INSTRUMENT_CHARACTERISTICS
)

@dataclass
class IsanMusicConfig:
    """Configuration for Isan Music Generation"""
    # Model architecture
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    warmup_steps: int = 4000
    
    # Audio parameters
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    
    # Isan-specific parameters
    instruments: List[str] = None
    cultural_weights: Dict[str, float] = None
    rhythm_patterns: List[str] = None
    
    def __post_init__(self):
        if self.instruments is None:
            self.instruments = list(INSTRUMENT_CHARACTERISTICS.keys())
        if self.cultural_weights is None:
            self.cultural_weights = {
                'melodic_preservation': 1.0,
                'rhythmic_authenticity': 0.8,
                'timbral_characteristic': 0.9,
                'cultural_context': 0.7
            }
        if self.rhythm_patterns is None:
            self.rhythm_patterns = ['สามช่า', 'สองช่า', 'ลาวกระทบไม้', 'เต้ยมอญ']

class IsanMusicTransformer(nn.Module):
    """Transformer architecture for Isan music generation"""
    
    def __init__(self, config: IsanMusicConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Linear(config.n_mels, config.hidden_dim)
        self.positional_encoding = PositionalEncoding(config.hidden_dim, config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Instrument-specific heads
        self.instrument_heads = nn.ModuleDict({
            instrument: nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.n_mels)
            )
            for instrument in config.instruments
        })
        
        # Cultural context layer
        self.cultural_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, len(config.instruments))
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        instrument: str,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        x: [batch_size, seq_len, n_mels]
        """
        # Input embedding
        x = self.input_embedding(x)  # [batch_size, seq_len, hidden_dim]
        x = self.positional_encoding(x)
        
        # Transpose for transformer: [seq_len, batch_size, hidden_dim]
        x = x.transpose(0, 1)
        
        # Transformer encoding
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        
        # Transpose back: [batch_size, seq_len, hidden_dim]
        x = x.transpose(0, 1)
        
        # Instrument-specific head
        if instrument in self.instrument_heads:
            output = self.instrument_heads[instrument](x)
        else:
            # Use shared head for unknown instruments
            output = self.instrument_heads[list(self.instrument_heads.keys())[0]](x)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class IsanMusicGenerator(pl.LightningModule):
    """PyTorch Lightning module for Isan music generation"""
    
    def __init__(self, config: IsanMusicConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config))
        
        # Model components
        self.model = IsanMusicTransformer(config)
        self.preprocessor = IsanMusicPreprocessor(config.sample_rate)
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.cultural_loss = CulturalLoss(config.cultural_weights)
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def forward(
        self, 
        x: torch.Tensor, 
        instrument: str,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(x, instrument, mask)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        features = batch['features']
        audio = batch['audio']
        instrument = batch['instrument'][0]  # All items in batch should be same instrument
        
        # Forward pass
        generated = self(features, instrument)
        
        # Compute losses
        recon_loss = self.reconstruction_loss(generated, features)
        percept_loss = self.perceptual_loss(generated, features, instrument)
        cultural_loss = self.cultural_loss(generated, features, instrument)
        
        # Total loss with cultural weighting
        total_loss = (
            recon_loss + 
            0.5 * percept_loss + 
            0.3 * cultural_loss
        )
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('recon_loss', recon_loss)
        self.log('percept_loss', percept_loss)
        self.log('cultural_loss', cultural_loss)
        
        self.train_losses.append(total_loss.item())
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        features = batch['features']
        audio = batch['audio']
        instrument = batch['instrument'][0]
        
        # Forward pass
        generated = self(features, instrument)
        
        # Compute validation loss
        val_loss = self.reconstruction_loss(generated, features)
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.val_losses.append(val_loss.item())
        
        return val_loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def generate_music(
        self,
        instrument: str,
        seed_length: int = 100,
        generation_length: int = 500,
        temperature: float = 0.8
    ) -> torch.Tensor:
        """Generate music for specific instrument"""
        self.eval()
        
        with torch.no_grad():
            # Initialize with random seed
            seed = torch.randn(1, seed_length, self.config.n_mels)
            generated = seed
            
            # Generate sequence
            for i in range(0, generation_length, seed_length):
                # Forward pass
                output = self(generated, instrument)
                
                # Apply temperature
                output = output / temperature
                
                # Append to generated sequence
                generated = torch.cat([generated, output], dim=1)
                
                # Keep manageable length
                if generated.size(1) > generation_length:
                    break
            
            # Trim to desired length
            generated = generated[:, :generation_length, :]
            
        return generated

class PerceptualLoss(nn.Module):
    """Perceptual loss for music generation"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        generated: torch.Tensor, 
        target: torch.Tensor,
        instrument: str
    ) -> torch.Tensor:
        """Compute perceptual loss"""
        # Spectral loss
        spec_loss = F.mse_loss(generated, target)
        
        # Temporal consistency loss
        temporal_loss = F.mse_loss(
            generated[:, 1:, :] - generated[:, :-1, :],
            target[:, 1:, :] - target[:, :-1, :]
        )
        
        return spec_loss + 0.1 * temporal_loss

class CulturalLoss(nn.Module):
    """Cultural preservation loss for Isan music"""
    
    def __init__(self, cultural_weights: Dict[str, float]):
        super().__init__()
        self.weights = cultural_weights
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        instrument: str
    ) -> torch.Tensor:
        """Compute cultural loss"""
        # Instrument-specific cultural constraints
        cultural_constraint = 0.0
        
        if instrument == 'พิณ':
            # Emphasize melodic continuity for phin
            cultural_constraint += self._melodic_continuity_loss(generated)
        elif instrument == 'แคน':
            # Emphasize harmonic richness for khaen
            cultural_constraint += self._harmonic_richness_loss(generated)
        elif instrument == 'โหวด':
            # Emphasize rhythmic precision for woad
            cultural_constraint += self._rhythmic_precision_loss(generated)
        elif instrument == 'โปงลาง':
            # Emphasize percussive clarity for ponglang
            cultural_constraint += self._percussive_clarity_loss(generated)
        
        return cultural_constraint
    
    def _melodic_continuity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Encourage smooth melodic transitions"""
        diff = x[:, 1:, :] - x[:, :-1, :]
        return torch.mean(torch.abs(diff))
    
    def _harmonic_richness_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Encourage harmonic richness"""
        # Promote frequency diversity
        freq_variety = torch.var(x, dim=-1)
        return -torch.mean(freq_variety)  # Negative to encourage variety
    
    def _rhythmic_precision_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Encourage rhythmic precision"""
        # Promote clear onsets
        onset_strength = torch.abs(x[:, 1:, :] - x[:, :-1, :])
        return -torch.mean(onset_strength)
    
    def _percussive_clarity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Encourage percussive clarity"""
        # Promote sharp attacks
        attack_sharpness = torch.abs(x[:, 1:, :] - x[:, :-1, :])
        return -torch.mean(attack_sharpness)

class IsanMusicTrainer:
    """Trainer class for Isan music generation"""
    
    def __init__(self, config: IsanMusicConfig):
        self.config = config
        self.model = IsanMusicGenerator(config)
        self.data_module = None
    
    def prepare_data(
        self,
        data_path: str,
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> None:
        """Prepare data for training"""
        self.data_module = IsanMusicDataModule(
            data_path=data_path,
            config=self.config,
            train_split=train_split,
            val_split=val_split
        )
    
    def train(self, gpus: int = 0, max_epochs: Optional[int] = None) -> None:
        """Train the model"""
        if max_epochs is None:
            max_epochs = self.config.num_epochs
        
        # Trainer configuration
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpus if torch.cuda.is_available() else 0,
            precision=16 if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0,
            accumulate_grad_batches=4,
            log_every_n_steps=10
        )
        
        # Train
        trainer.fit(self.model, self.data_module)
    
    def save_model(self, path: str) -> None:
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'training_date': datetime.now().isoformat()
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {path}")

class IsanMusicDataModule(pl.LightningDataModule):
    """Data module for PyTorch Lightning"""
    
    def __init__(
        self,
        data_path: str,
        config: IsanMusicConfig,
        train_split: float = 0.8,
        val_split: float = 0.1
    ):
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.train_split = train_split
        self.val_split = val_split
        self.datasets = {}
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        for instrument in self.config.instruments:
            dataset = IsanMusicDataset(
                data_path=self.data_path,
                instrument=instrument,
                sample_rate=self.config.sample_rate,
                duration=10.0
            )
            
            # Split dataset
            total_size = len(dataset)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            self.datasets[instrument] = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader"""
        # Combine all instruments for training
        train_datasets = []
        for instrument, splits in self.datasets.items():
            train_datasets.append(splits[0])  # Train split
        
        combined_dataset = torch.utils.data.ConcatDataset(train_datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        val_datasets = []
        for instrument, splits in self.datasets.items():
            val_datasets.append(splits[1])  # Validation split
        
        combined_dataset = torch.utils.data.ConcatDataset(val_datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

# Example usage
if __name__ == "__main__":
    print("🎵 เริ่มต้นการสร้างดนตรีอีสานด้วย AI")
    
    # Configuration
    config = IsanMusicConfig(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        batch_size=8,
        num_epochs=50
    )
    
    print(f"⚙️  ค่าตั้งค่า: {config}")
    
    # Initialize trainer
    trainer = IsanMusicTrainer(config)
    
    # Example training (commented out to avoid running in demo)
    # trainer.prepare_data("./isan_audio_data")
    # trainer.train()
    # trainer.save_model("./isan_music_model.pt")
    
    print("✅ โมเดลพร้อมสำหรับการฝึกสอน")