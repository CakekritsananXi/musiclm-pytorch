"""
Performance Optimization Utilities for Isan Music Generation
เครื่องมือเพิ่มประสิทธิภาพสำหรับการสร้างดนตรีอีสาน
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any, List
import time
from functools import wraps
from contextlib import contextmanager
import numpy as np


class PerformanceOptimizer:
    """Utilities for optimizing model performance"""
    
    @staticmethod
    def enable_mixed_precision(model: nn.Module) -> nn.Module:
        """
        Enable mixed precision (FP16) for faster inference
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        if torch.cuda.is_available():
            model = model.half()
            print("✅ Mixed precision (FP16) enabled")
        else:
            print("⚠️  CUDA not available, keeping FP32")
        return model
    
    @staticmethod
    def compile_model(model: nn.Module, mode: str = 'default') -> nn.Module:
        """
        Compile model using PyTorch 2.0+ compilation
        
        Args:
            model: PyTorch model to compile
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            
        Returns:
            Compiled model
        """
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode=mode)
                print(f"✅ Model compiled with mode: {mode}")
            except Exception as e:
                print(f"⚠️  Compilation failed: {e}")
        else:
            print("⚠️  torch.compile not available (requires PyTorch 2.0+)")
        return model
    
    @staticmethod
    def optimize_for_inference(
        model: nn.Module,
        use_fp16: bool = True,
        use_compile: bool = True,
        compile_mode: str = 'reduce-overhead'
    ) -> nn.Module:
        """
        Apply multiple optimizations for inference
        
        Args:
            model: PyTorch model to optimize
            use_fp16: Enable mixed precision
            use_compile: Enable torch.compile
            compile_mode: Compilation mode
            
        Returns:
            Optimized model
        """
        model.eval()
        
        if use_fp16:
            model = PerformanceOptimizer.enable_mixed_precision(model)
        
        if use_compile:
            model = PerformanceOptimizer.compile_model(model, mode=compile_mode)
        
        return model
    
    @staticmethod
    def benchmark_inference(
        model: nn.Module,
        input_generator: Callable,
        num_runs: int = 100,
        warmup_runs: int = 10,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        Benchmark model inference performance
        
        Args:
            model: Model to benchmark
            input_generator: Function that generates model inputs
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            device: Device to run on
            
        Returns:
            Dictionary with benchmark results
        """
        model = model.to(device)
        model.eval()
        
        # Warmup
        print(f"🔥 Warming up ({warmup_runs} runs)...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                inputs = input_generator()
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                elif isinstance(inputs, (tuple, list)):
                    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
                _ = model(*inputs) if isinstance(inputs, (tuple, list)) else model(inputs)
        
        # Benchmark
        print(f"⏱️  Benchmarking ({num_runs} runs)...")
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                inputs = input_generator()
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                elif isinstance(inputs, (tuple, list)):
                    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
                
                start = time.perf_counter()
                _ = model(*inputs) if isinstance(inputs, (tuple, list)) else model(inputs)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append(end - start)
        
        # Calculate statistics
        times = np.array(times) * 1000  # Convert to ms
        
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'median_time_ms': float(np.median(times)),
            'p95_time_ms': float(np.percentile(times, 95)),
            'p99_time_ms': float(np.percentile(times, 99)),
            'throughput_samples_per_sec': 1000.0 / np.mean(times)
        }
        
        return results
    
    @staticmethod
    def profile_model(
        model: nn.Module,
        input_data: Any,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Profile model to identify bottlenecks
        
        Args:
            model: Model to profile
            input_data: Input data for profiling
            device: Device to run on
            
        Returns:
            Profiling results
        """
        model = model.to(device)
        model.eval()
        
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        total_memory_mb = (param_memory + buffer_memory) / (1024 ** 2)
        
        # Run forward pass and measure time
        with torch.no_grad():
            start = time.perf_counter()
            _ = model(input_data)
            if device == 'cuda':
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - start
        
        results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_memory_mb': total_memory_mb,
            'forward_pass_time_ms': forward_time * 1000,
        }
        
        return results


class CachedSynthesizer:
    """Cached version of synthesizer for improved performance"""
    
    def __init__(self, synthesizer, cache_size: int = 1000):
        """
        Initialize cached synthesizer
        
        Args:
            synthesizer: Underlying synthesizer
            cache_size: Maximum cache size
        """
        self.synthesizer = synthesizer
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def synthesize_note(
        self,
        frequency: float,
        duration: float,
        velocity: float = 0.8,
        **kwargs
    ) -> torch.Tensor:
        """
        Synthesize note with caching
        
        Args:
            frequency: Note frequency in Hz
            duration: Duration in seconds
            velocity: Note velocity (0-1)
            
        Returns:
            Synthesized audio
        """
        # Create cache key
        cache_key = (
            round(frequency, 2),
            round(duration, 3),
            round(velocity, 2)
        )
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key].clone()
        
        # Generate and cache
        self.cache_misses += 1
        audio = self.synthesizer.synthesize_note(frequency, duration, velocity, **kwargs)
        
        # Manage cache size
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[cache_key] = audio.clone()
        return audio
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class BatchProcessor:
    """Efficient batch processing for audio generation"""
    
    @staticmethod
    def batch_synthesize_notes(
        synthesizer,
        notes: List[Dict],
        batch_size: int = 32
    ) -> List[torch.Tensor]:
        """
        Batch process multiple notes efficiently
        
        Args:
            synthesizer: Synthesizer instance
            notes: List of note dictionaries
            batch_size: Batch size for processing
            
        Returns:
            List of synthesized audio tensors
        """
        results = []
        
        for i in range(0, len(notes), batch_size):
            batch = notes[i:i + batch_size]
            batch_results = [
                synthesizer.synthesize_note(**note)
                for note in batch
            ]
            results.extend(batch_results)
        
        return results
    
    @staticmethod
    def parallel_generate(
        generator,
        prompts: List[str],
        num_workers: int = 4,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generate audio for multiple prompts in parallel
        
        Args:
            generator: Generator instance
            prompts: List of text prompts
            num_workers: Number of parallel workers
            kwargs: Additional generation parameters
            
        Returns:
            List of generated audio tensors
        """
        # Note: This is a simplified version
        # For true parallelism, consider using multiprocessing or ray
        results = []
        for prompt in prompts:
            audio = generator.generate_from_text(prompt, **kwargs)
            results.append(audio)
        
        return results


@contextmanager
def performance_timer(name: str = "Operation"):
    """Context manager for timing operations"""
    print(f"⏱️  Starting: {name}")
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    duration = (end - start) * 1000
    print(f"✅ Completed: {name} ({duration:.2f}ms)")


def memory_efficient_generation(
    generator,
    text: str,
    total_duration: float,
    chunk_duration: float = 30.0,
    overlap: float = 1.0,
    **kwargs
) -> torch.Tensor:
    """
    Generate long audio in chunks to save memory
    
    Args:
        generator: Generator instance
        text: Text prompt
        total_duration: Total audio duration
        chunk_duration: Duration of each chunk
        overlap: Overlap between chunks for smooth transitions
        kwargs: Additional generation parameters
        
    Returns:
        Complete audio tensor
    """
    chunks = []
    num_chunks = int(np.ceil(total_duration / chunk_duration))
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration + overlap, total_duration)
        chunk_dur = end_time - start_time
        
        print(f"Generating chunk {i+1}/{num_chunks} ({start_time:.1f}s - {end_time:.1f}s)")
        
        chunk = generator.generate_from_text(
            text=text,
            duration=chunk_dur,
            seed=i,  # Maintain some consistency
            **kwargs
        )
        
        # Apply fade at boundaries for smooth transitions
        if i > 0:
            # Fade in at start
            fade_samples = int(overlap * generator.config.sample_rate)
            fade_in = torch.linspace(0, 1, fade_samples).unsqueeze(0)
            chunk[..., :fade_samples] *= fade_in
        
        if i < num_chunks - 1:
            # Fade out at end
            fade_samples = int(overlap * generator.config.sample_rate)
            fade_out = torch.linspace(1, 0, fade_samples).unsqueeze(0)
            chunk[..., -fade_samples:] *= fade_out
        
        chunks.append(chunk)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate chunks with overlap handling
    result_chunks = [chunks[0]]
    for i in range(1, len(chunks)):
        overlap_samples = int(overlap * generator.config.sample_rate)
        # Remove overlap from previous chunk and current chunk start
        result_chunks[-1] = result_chunks[-1][..., :-overlap_samples]
        result_chunks.append(chunks[i][..., overlap_samples:])
    
    return torch.cat(result_chunks, dim=-1)


class GPUMemoryManager:
    """Manage GPU memory efficiently"""
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'free_gb': reserved - allocated
        }
    
    @staticmethod
    def print_memory_stats():
        """Print GPU memory statistics"""
        stats = GPUMemoryManager.get_memory_stats()
        if stats:
            print("🎮 GPU Memory:")
            print(f"   Allocated: {stats['allocated_gb']:.2f} GB")
            print(f"   Reserved:  {stats['reserved_gb']:.2f} GB")
            print(f"   Free:      {stats['free_gb']:.2f} GB")
            print(f"   Max Used:  {stats['max_allocated_gb']:.2f} GB")
    
    @contextmanager
    def memory_tracker(self):
        """Context manager to track memory usage"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
        
        yield
        
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            
            print(f"📊 Memory Usage:")
            print(f"   Start:  {start_mem / (1024**3):.2f} GB")
            print(f"   End:    {end_mem / (1024**3):.2f} GB")
            print(f"   Peak:   {peak_mem / (1024**3):.2f} GB")
            print(f"   Delta:  {(end_mem - start_mem) / (1024**3):.2f} GB")


def optimize_dataloader(
    dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2
):
    """
    Create optimized DataLoader
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch
        
    Returns:
        Optimized DataLoader
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )


# Decorator for timing functions
def time_function(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = (end - start) * 1000
        print(f"⏱️  {func.__name__}: {duration:.2f}ms")
        return result
    return wrapper


if __name__ == '__main__':
    # Example usage
    print("Performance Optimization Utilities for Isan Music Generation")
    print("=" * 70)
    
    # Test GPU memory manager
    if torch.cuda.is_available():
        manager = GPUMemoryManager()
        manager.print_memory_stats()
    else:
        print("⚠️  CUDA not available, running on CPU")
