"""
🎵 Audio Quality Validation and Silence Detection for Higgs-Audio Training

Provides comprehensive debugging capabilities for audio generation issues:
- Silence detection and analysis following silence-detection-debug patterns
- Audio quality metrics validation 
- Reference audio conditioning verification
- Token sequence integrity checking
- Audio generation termination validation
- DualFFN architecture compatibility

Based on project specifications and arb_inference.py patterns.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import json

# 🏠 ENHANCED: Robust import system for root directory execution
current_file = Path(__file__).resolve()
trainer_dir = current_file.parent
higgs_audio_root = trainer_dir.parent

if str(higgs_audio_root) not in sys.path:
    sys.path.insert(0, str(higgs_audio_root))

# 🔧 Conditional imports for ML dependencies
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for non-ML operations
    class torch:
        class Tensor:
            pass
    np = None

# 🔧 Conditional imports for dependencies
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


class AudioQualityValidator:
    """
    🔍 Comprehensive audio quality validation for Higgs-Audio training debugging.
    
    Provides detailed analysis of audio generation issues, particularly
    silence generation problems and DualFFN architecture validation.
    
    Features:
    - Silence detection and pattern analysis
    - Audio quality metrics computation
    - Token sequence validation for 12-codebook DAC
    - Dynamic range analysis
    - Spectral analysis for audio quality
    - Comprehensive reporting system
    """
    
    def __init__(
        self,
        silence_threshold: float = 1e-6,
        min_audio_length: float = 1.0,  # seconds
        max_audio_length: float = 30.0,  # seconds  
        max_silence_ratio: float = 0.3,
        target_sample_rate: int = 24000,
        expected_codebooks: int = 12  # DAC standard
    ):
        """
        Initialize audio quality validator with Higgs-Audio specifications.
        
        Args:
            silence_threshold: Energy threshold below which audio is considered silence
            min_audio_length: Minimum acceptable audio length in seconds
            max_audio_length: Maximum acceptable audio length in seconds
            max_silence_ratio: Maximum ratio of silence allowed in audio
            target_sample_rate: Expected output sample rate (24kHz for Higgs-Audio)
            expected_codebooks: Expected number of audio codebooks for DAC
        """
        self.silence_threshold = silence_threshold
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.max_silence_ratio = max_silence_ratio
        self.target_sample_rate = target_sample_rate
        self.expected_codebooks = expected_codebooks
        
        self.validation_results = []
        
        # Validate dependencies
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch not available - audio validation will be limited")
    
    def validate_audio_waveform(
        self,
        waveform: Union[torch.Tensor, "np.ndarray", None],
        sample_rate: int,
        audio_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        🎵 Comprehensive validation of generated audio waveform.
        
        Validates audio following Higgs-Audio quality standards and detects
        common generation issues like silence, clipping, and poor dynamic range.
        
        Args:
            waveform: Audio waveform as numpy array or torch tensor
            sample_rate: Sample rate of the audio
            audio_id: Identifier for this audio sample
            
        Returns:
            Dictionary containing validation results and metrics
        """
        results = {
            'audio_id': audio_id,
            'sample_rate': sample_rate,
            'duration': 0.0,
            'valid': True,
            'issues': [],
            'metrics': {},
            'validation_type': 'waveform'
        }
        
        # Check if ML dependencies are available
        if not TORCH_AVAILABLE:
            results['issues'].append('PyTorch not available for audio validation')
            results['valid'] = False
            return results
        
        # Handle None input
        if waveform is None:
            results['issues'].append('Waveform is None')
            results['valid'] = False
            return results
        
        # Convert to numpy if torch tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        
        # Ensure waveform is 1D
        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()
        
        # Calculate duration
        if len(waveform) > 0:
            results['duration'] = len(waveform) / sample_rate
        
        # 1. Basic waveform validation
        self._validate_basic_properties(waveform, results)
        
        # 2. Silence detection and analysis
        self._analyze_silence(waveform, sample_rate, results)
        
        # 3. Audio quality metrics
        self._compute_quality_metrics(waveform, sample_rate, results)
        
        # 4. Duration validation
        self._validate_duration(results)
        
        # 5. Dynamic range analysis
        self._analyze_dynamic_range(waveform, results)
        
        # 6. Sample rate validation
        self._validate_sample_rate(sample_rate, results)
        
        # Log results
        self._log_validation_results(results)
        
        # Store for batch analysis
        self.validation_results.append(results)
        
        return results
    
    def _validate_basic_properties(self, waveform: "np.ndarray", results: Dict[str, Any]):
        """🔍 Validate basic waveform properties for corruption detection."""
        if not TORCH_AVAILABLE:
            return
        
        # Check for NaN or infinite values
        if np.isnan(waveform).any():
            results['issues'].append('Contains NaN values')
            results['valid'] = False
        
        if np.isinf(waveform).any():
            results['issues'].append('Contains infinite values')
            results['valid'] = False
        
        # Check waveform amplitude range
        if len(waveform) > 0:
            max_amplitude = np.abs(waveform).max()
            results['metrics']['max_amplitude'] = float(max_amplitude)
            
            if max_amplitude > 1.0:
                results['issues'].append(f'Amplitude clipping detected: {max_amplitude:.3f}')
                results['valid'] = False
            elif max_amplitude < 1e-8:
                results['issues'].append('Extremely low amplitude (near silence)')
                results['valid'] = False
            
            # RMS amplitude
            rms_amplitude = np.sqrt(np.mean(waveform ** 2))
            results['metrics']['rms_amplitude'] = float(rms_amplitude)
        else:
            results['issues'].append('Empty waveform')
            results['valid'] = False
    
    def _analyze_silence(self, waveform: "np.ndarray", sample_rate: int, results: Dict[str, Any]):
        """🔇 Analyze silence patterns following silence-detection-debug patterns."""
        if not TORCH_AVAILABLE or len(waveform) == 0:
            return
        
        # Compute RMS energy in sliding windows
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = window_size // 2
        
        if window_size >= len(waveform):
            # Handle short audio
            energy = np.sqrt(np.mean(waveform ** 2))
            energies = np.array([energy])
        else:
            energies = []
            for i in range(0, len(waveform) - window_size + 1, hop_size):
                window = waveform[i:i + window_size]
                energy = np.sqrt(np.mean(window ** 2))
                energies.append(energy)
            energies = np.array(energies)
        
        # Detect silence segments
        silence_mask = energies < self.silence_threshold
        silence_ratio = silence_mask.sum() / len(silence_mask) if len(silence_mask) > 0 else 0.0
        
        results['metrics']['silence_ratio'] = float(silence_ratio)
        results['metrics']['avg_energy'] = float(energies.mean())
        results['metrics']['min_energy'] = float(energies.min())
        results['metrics']['max_energy'] = float(energies.max())
        results['metrics']['num_energy_windows'] = len(energies)
        
        # Check for problematic silence patterns
        if silence_ratio > self.max_silence_ratio:
            results['issues'].append(f'Excessive silence: {silence_ratio:.1%} (threshold: {self.max_silence_ratio:.1%})')
            results['valid'] = False
        
        # Check for complete silence
        if silence_ratio > 0.95:
            results['issues'].append('Audio is almost completely silent')
            results['valid'] = False
        
        # Analyze silence distribution
        silence_segments = self._find_silence_segments(silence_mask, sample_rate, hop_size)
        results['metrics']['silence_segments'] = silence_segments
        results['metrics']['num_silence_segments'] = len(silence_segments)
        
        # Check for extended silence at the end (common generation issue)
        if len(silence_segments) > 0:
            last_segment = silence_segments[-1]
            total_duration = len(waveform) / sample_rate
            if last_segment['duration'] > total_duration * 0.3:
                results['issues'].append(f'Extended silence at end: {last_segment["duration"]:.1f}s')
    
    def _find_silence_segments(self, silence_mask: "np.ndarray", sample_rate: int, hop_size: int) -> List[Dict]:
        """🔍 Find continuous silence segments for pattern analysis."""
        if not TORCH_AVAILABLE:
            return []
        
        segments = []
        in_silence = False
        start_idx = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                # Start of silence segment
                in_silence = True
                start_idx = i
            elif not is_silent and in_silence:
                # End of silence segment
                in_silence = False
                duration = (i - start_idx) * hop_size / sample_rate
                segments.append({
                    'start_time': start_idx * hop_size / sample_rate,
                    'duration': duration
                })
        
        # Handle case where audio ends in silence
        if in_silence:
            duration = (len(silence_mask) - start_idx) * hop_size / sample_rate
            segments.append({
                'start_time': start_idx * hop_size / sample_rate,
                'duration': duration
            })
        
        return segments
    
    def _compute_quality_metrics(self, waveform: "np.ndarray", sample_rate: int, results: Dict[str, Any]):
        """📊 Compute comprehensive audio quality metrics."""
        if not TORCH_AVAILABLE or len(waveform) == 0:
            return
        
        # Signal power
        signal_power = np.mean(waveform ** 2)
        results['metrics']['signal_power'] = float(signal_power)
        
        # Zero crossing rate
        zcr = np.sum(np.diff(np.signbit(waveform))) / len(waveform)
        results['metrics']['zero_crossing_rate'] = float(zcr)
        
        # Spectral analysis
        if len(waveform) > 1024:
            # Use power of 2 for efficient FFT
            fft_size = min(2048, 2 ** int(np.log2(len(waveform))))
            stft = np.abs(np.fft.fft(waveform[:fft_size]))
            freqs = np.fft.fftfreq(fft_size, 1/sample_rate)
            
            # Only use positive frequencies
            positive_freqs = freqs[:fft_size//2]
            positive_stft = stft[:fft_size//2]
            
            if positive_stft.sum() > 0:
                # Spectral centroid
                spectral_centroid = np.sum(positive_freqs * positive_stft) / positive_stft.sum()
                results['metrics']['spectral_centroid'] = float(spectral_centroid)
                
                # Spectral bandwidth
                spectral_bandwidth = np.sqrt(np.sum(((positive_freqs - spectral_centroid) ** 2) * positive_stft) / positive_stft.sum())
                results['metrics']['spectral_bandwidth'] = float(spectral_bandwidth)
                
                # High frequency content (above 8kHz)
                high_freq_mask = positive_freqs > 8000
                if high_freq_mask.any():
                    high_freq_energy = positive_stft[high_freq_mask].sum()
                    total_energy = positive_stft.sum()
                    high_freq_ratio = high_freq_energy / total_energy
                    results['metrics']['high_freq_ratio'] = float(high_freq_ratio)
                else:
                    results['metrics']['high_freq_ratio'] = 0.0
                
                # Low frequency content (below 1kHz)
                low_freq_mask = positive_freqs < 1000
                if low_freq_mask.any():
                    low_freq_energy = positive_stft[low_freq_mask].sum()
                    low_freq_ratio = low_freq_energy / total_energy
                    results['metrics']['low_freq_ratio'] = float(low_freq_ratio)
    
    def _validate_duration(self, results: Dict[str, Any]):
        """⏱️ Validate audio duration against Higgs-Audio standards."""
        duration = results['duration']
        
        if duration < self.min_audio_length:
            results['issues'].append(f'Audio too short: {duration:.1f}s (min: {self.min_audio_length}s)')
            results['valid'] = False
        elif duration > self.max_audio_length:
            results['issues'].append(f'Audio too long: {duration:.1f}s (max: {self.max_audio_length}s)')
            results['valid'] = False
            
        results['metrics']['duration_valid'] = self.min_audio_length <= duration <= self.max_audio_length
    
    def _validate_sample_rate(self, sample_rate: int, results: Dict[str, Any]):
        """🎛️ Validate sample rate against Higgs-Audio target (24kHz)."""
        results['metrics']['sample_rate_match'] = sample_rate == self.target_sample_rate
        
        if sample_rate != self.target_sample_rate:
            results['issues'].append(f'Sample rate mismatch: {sample_rate}Hz (expected: {self.target_sample_rate}Hz)')
    
    def _analyze_dynamic_range(self, waveform: "np.ndarray", results: Dict[str, Any]):
        """📈 Analyze dynamic range for audio quality assessment."""
        if not TORCH_AVAILABLE or len(waveform) == 0:
            return
            
        # RMS levels in dB
        rms = np.sqrt(np.mean(waveform ** 2))
        if rms > 0:
            rms_db = 20 * np.log10(rms)
            results['metrics']['rms_db'] = float(rms_db)
            
            # Peak levels
            peak = np.abs(waveform).max()
            if peak > 0:
                peak_db = 20 * np.log10(peak)
                crest_factor = peak_db - rms_db
                results['metrics']['peak_db'] = float(peak_db)
                results['metrics']['crest_factor'] = float(crest_factor)
                
                # Check for poor dynamic range
                if crest_factor < 3:
                    results['issues'].append(f'Low dynamic range: {crest_factor:.1f}dB crest factor')
                
                # Check for over-compression
                if crest_factor > 30:
                    results['issues'].append(f'Excessive dynamic range: {crest_factor:.1f}dB crest factor')
    
    def _log_validation_results(self, results: Dict[str, Any]):
        """📝 Log validation results with appropriate severity levels."""
        audio_id = results['audio_id']
        
        if results['valid']:
            logger.info(f"✅ Audio {audio_id}: Valid ({results['duration']:.1f}s, {results['metrics'].get('silence_ratio', 0):.1%} silence)")
        else:
            logger.error(f"❌ Audio {audio_id}: Invalid - {', '.join(results['issues'])}")
            
        # Log detailed metrics in debug mode
        if logger.level == "DEBUG":
            logger.debug(f"📊 Audio {audio_id} metrics: {json.dumps(results['metrics'], indent=2)}")
    
    def validate_audio_tokens(
        self,
        audio_tokens: Optional[torch.Tensor],
        audio_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        🎵 Validate audio token sequence for DAC generation issues.
        
        Validates audio tokens following Higgs-Audio DAC patterns with
        12 codebooks and proper token diversity analysis.
        
        Args:
            audio_tokens: Audio tokens tensor [num_codebooks, seq_len]
            audio_id: Identifier for this audio
            
        Returns:
            Dictionary containing token validation results
        """
        results = {
            'audio_id': audio_id,
            'valid': True,
            'issues': [],
            'metrics': {},
            'validation_type': 'tokens'
        }
        
        if not TORCH_AVAILABLE:
            results['issues'].append('PyTorch not available for token validation')
            results['valid'] = False
            return results
        
        if audio_tokens is None or audio_tokens.numel() == 0:
            results['issues'].append('Empty or None audio tokens')
            results['valid'] = False
            return results
        
        # Token sequence analysis
        if len(audio_tokens.shape) != 2:
            results['issues'].append(f'Invalid token shape: {audio_tokens.shape} (expected [num_codebooks, seq_len])')
            results['valid'] = False
            return results
        
        num_codebooks, seq_len = audio_tokens.shape
        results['metrics']['num_codebooks'] = num_codebooks
        results['metrics']['sequence_length'] = seq_len
        
        # Check for expected codebook count (DAC standard)
        if num_codebooks != self.expected_codebooks:
            results['issues'].append(f'Unexpected codebook count: {num_codebooks} (expected {self.expected_codebooks})')
        
        # Token diversity analysis
        unique_tokens_per_codebook = []
        token_ranges = []
        
        for cb in range(num_codebooks):
            unique_tokens = torch.unique(audio_tokens[cb]).numel()
            unique_tokens_per_codebook.append(unique_tokens)
            
            # Token value range
            min_token = audio_tokens[cb].min().item()
            max_token = audio_tokens[cb].max().item()
            token_ranges.append((min_token, max_token))
        
        avg_unique_tokens = float(np.mean(unique_tokens_per_codebook))
        results['metrics']['avg_unique_tokens'] = avg_unique_tokens
        results['metrics']['min_unique_tokens'] = int(min(unique_tokens_per_codebook))
        results['metrics']['max_unique_tokens'] = int(max(unique_tokens_per_codebook))
        results['metrics']['token_ranges'] = token_ranges
        
        # Check for low diversity (indication of generation issues)
        diversity_threshold = max(10, seq_len * 0.1)  # At least 10 or 10% of sequence
        if avg_unique_tokens < diversity_threshold:
            results['issues'].append(f'Low token diversity: {avg_unique_tokens:.1f} unique tokens per codebook (threshold: {diversity_threshold})')
            results['valid'] = False
        
        # Check for proper token ranges (DAC typically uses 0-1023)
        for cb, (min_tok, max_tok) in enumerate(token_ranges):
            if min_tok < 0 or max_tok > 1023:
                results['issues'].append(f'Codebook {cb}: Token range [{min_tok}, {max_tok}] outside expected [0, 1023]')
        
        # Check for repeated patterns (common in failed generation)
        self._analyze_token_patterns(audio_tokens, results)
        
        logger.debug(f"🎵 Token validation {audio_id}: {seq_len} tokens, {avg_unique_tokens:.1f} avg unique")
        
        # Store for batch analysis
        self.validation_results.append(results)
        
        return results
    
    def _analyze_token_patterns(self, audio_tokens: torch.Tensor, results: Dict[str, Any]):
        """🔍 Analyze for problematic token patterns in generation."""
        if not TORCH_AVAILABLE:
            return
        
        num_codebooks, seq_len = audio_tokens.shape
        
        # Check for excessive repetition in each codebook
        for cb in range(min(num_codebooks, 8)):  # Check first 8 codebooks
            tokens = audio_tokens[cb]
            
            # Simple repetition detection
            if seq_len > 10:
                # Check for runs of identical tokens
                diff = torch.diff(tokens)
                zero_diff_count = (diff == 0).sum().item()
                repetition_ratio = zero_diff_count / (seq_len - 1)
                
                if repetition_ratio > 0.8:
                    results['issues'].append(f'Codebook {cb}: Excessive repetition ({repetition_ratio:.1%})')
                    results['valid'] = False
                
                # Check for alternating patterns (another common issue)
                if seq_len > 20:
                    # Simple alternating detection
                    alternating_count = 0
                    for i in range(2, seq_len - 2):
                        if tokens[i] == tokens[i-2] and tokens[i+1] == tokens[i-1] and tokens[i] != tokens[i+1]:
                            alternating_count += 1
                    
                    alternating_ratio = alternating_count / (seq_len - 4)
                    if alternating_ratio > 0.5:
                        results['issues'].append(f'Codebook {cb}: Excessive alternating pattern ({alternating_ratio:.1%})')
    
    def generate_validation_report(self, output_path: str):
        """📊 Generate comprehensive validation report for training analysis."""
        if not self.validation_results:
            logger.warning("No validation results to report")
            return
        
        # Aggregate statistics
        total_samples = len(self.validation_results)
        valid_samples = sum(1 for r in self.validation_results if r['valid'])
        
        # Separate by validation type
        waveform_results = [r for r in self.validation_results if r.get('validation_type') == 'waveform']
        token_results = [r for r in self.validation_results if r.get('validation_type') == 'tokens']
        
        report = {
            'summary': {
                'total_samples': total_samples,
                'valid_samples': valid_samples,
                'validation_rate': valid_samples / total_samples,
                'waveform_samples': len(waveform_results),
                'token_samples': len(token_results),
            },
            'common_issues': self._analyze_common_issues(),
            'quality_metrics': self._aggregate_quality_metrics(),
            'validation_settings': {
                'silence_threshold': self.silence_threshold,
                'min_audio_length': self.min_audio_length,
                'max_audio_length': self.max_audio_length,
                'max_silence_ratio': self.max_silence_ratio,
                'target_sample_rate': self.target_sample_rate,
                'expected_codebooks': self.expected_codebooks
            },
            'detailed_results': self.validation_results
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Validation report saved: {output_path}")
        logger.info(f"   Valid samples: {valid_samples}/{total_samples} ({valid_samples/total_samples:.1%})")
        
        # Log summary of common issues
        common_issues = self._analyze_common_issues()
        if common_issues:
            logger.info("🔍 Most common issues:")
            for issue, count in list(common_issues.items())[:5]:
                logger.info(f"   {issue}: {count} samples")
    
    def _analyze_common_issues(self) -> Dict[str, int]:
        """🔍 Analyze most common validation issues across samples."""
        issue_counts = {}
        
        for result in self.validation_results:
            for issue in result['issues']:
                # Normalize issue text for counting
                normalized_issue = issue.split(':')[0]  # Take part before colon
                issue_counts[normalized_issue] = issue_counts.get(normalized_issue, 0) + 1
        
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _aggregate_quality_metrics(self) -> Dict[str, float]:
        """📈 Aggregate quality metrics across all samples for analysis."""
        if not self.validation_results:
            return {}
        
        # Collect metrics
        metrics_lists = {}
        for result in self.validation_results:
            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    if metric not in metrics_lists:
                        metrics_lists[metric] = []
                    metrics_lists[metric].append(value)
        
        # Compute aggregated statistics
        aggregated = {}
        for metric, values in metrics_lists.items():
            if values and TORCH_AVAILABLE:
                aggregated[f'{metric}_mean'] = float(np.mean(values))
                aggregated[f'{metric}_std'] = float(np.std(values))
                aggregated[f'{metric}_min'] = float(np.min(values))
                aggregated[f'{metric}_max'] = float(np.max(values))
                aggregated[f'{metric}_count'] = len(values)
        
        return aggregated
    
    def reset_validation_results(self):
        """🔄 Reset validation results for new batch analysis."""
        self.validation_results = []
        logger.info("🔄 Validation results reset")


# 🌐 Global audio quality validator instance
audio_validator = AudioQualityValidator()


def validate_training_audio_sample(
    waveform: Optional[Union[torch.Tensor, "np.ndarray"]] = None,
    audio_tokens: Optional[torch.Tensor] = None,
    sample_rate: int = 24000,
    audio_id: str = "training_sample"
) -> Dict[str, Any]:
    """
    🎯 Convenience function for validating training audio samples.
    
    Args:
        waveform: Audio waveform for validation
        audio_tokens: Audio tokens for validation
        sample_rate: Sample rate of the audio
        audio_id: Identifier for logging
        
    Returns:
        Combined validation results
    """
    results = {
        'audio_id': audio_id,
        'waveform_valid': True,
        'tokens_valid': True,
        'overall_valid': True,
        'issues': []
    }
    
    # Validate waveform if provided
    if waveform is not None:
        waveform_results = audio_validator.validate_audio_waveform(waveform, sample_rate, audio_id)
        results['waveform_results'] = waveform_results
        results['waveform_valid'] = waveform_results['valid']
        if not waveform_results['valid']:
            results['issues'].extend([f"Waveform: {issue}" for issue in waveform_results['issues']])
    
    # Validate tokens if provided
    if audio_tokens is not None:
        token_results = audio_validator.validate_audio_tokens(audio_tokens, audio_id)
        results['token_results'] = token_results
        results['tokens_valid'] = token_results['valid']
        if not token_results['valid']:
            results['issues'].extend([f"Tokens: {issue}" for issue in token_results['issues']])
    
    # Overall validation
    results['overall_valid'] = results['waveform_valid'] and results['tokens_valid']
    
    return results