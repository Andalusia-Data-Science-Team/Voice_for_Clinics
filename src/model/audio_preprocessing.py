import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy import signal

class AudioPreprocessingService:
    """Service for audio preprocessing and enhancement."""
    
    @staticmethod
    def preprocess_audio(input_file_path, output_file_path=None, 
                        normalize=True, remove_noise=True, trim_silence=True,
                        apply_highpass=True, apply_lowpass=True):
        """
        Preprocess audio file to improve quality for transcription.
        
        Args:
            input_file_path: Path to the input audio file
            output_file_path: Path to save the processed audio file (if None, a temp file is created)
            normalize: Whether to normalize audio volume
            remove_noise: Whether to apply noise reduction
            trim_silence: Whether to trim silence from the beginning and end
            apply_highpass: Whether to apply high-pass filter (remove low frequencies)
            apply_lowpass: Whether to apply low-pass filter (remove high frequencies)
            
        Returns:
            Path to the processed audio file
        """
        # Create temp file if output path not provided
        if not output_file_path:
            temp_dir = tempfile.mkdtemp()
            output_file_path = os.path.join(temp_dir, "processed_audio.wav")
        
        # Load audio file using librosa
        try:
            # Check if input_file_path is None or empty
            if not input_file_path:
                raise ValueError("Input file path is None or empty")
            
            # Use pydub to convert any format to WAV first if needed
            if not input_file_path.lower().endswith('.wav'):
                audio = AudioSegment.from_file(input_file_path)
                temp_wav = os.path.join(os.path.dirname(input_file_path), "temp_conversion.wav")
                audio.export(temp_wav, format="wav")
                input_file_path = temp_wav
                
            # Load the audio with librosa
            y, sr = librosa.load(input_file_path, sr=None)
            
            # Apply preprocessing steps
            if trim_silence:
                # Trim leading and trailing silence
                y, _ = librosa.effects.trim(y, top_db=20)
            
            if apply_highpass:
                # Apply high-pass filter (300Hz cutoff to keep speech but remove some low rumble)
                b, a = signal.butter(5, 300/(sr/2), 'highpass')
                y = signal.filtfilt(b, a, y)
            
            if apply_lowpass:
                # Apply low-pass filter (8000Hz cutoff, most speech content is below this)
                b, a = signal.butter(5, 8000/(sr/2), 'lowpass')
                y = signal.filtfilt(b, a, y)
            
            if remove_noise:
                # Simple noise reduction using spectral gating
                # This is a simplified approach - for more advanced noise reduction, consider using librosa.decompose.nn_filter
                # or a dedicated library like noisereduce
                
                # Estimate noise from a small segment (assuming first 0.5 seconds might be noise/silence)
                noise_sample = y[:int(sr * 0.5)] if len(y) > sr * 0.5 else y[:int(len(y) * 0.1)]
                
                # Compute noise profile
                noise_stft = librosa.stft(noise_sample)
                noise_power = np.mean(np.abs(noise_stft)**2, axis=1)
                
                # Compute STFT of the signal
                speech_stft = librosa.stft(y)
                speech_power = np.abs(speech_stft)**2
                
                # Apply simple spectral subtraction with a floor
                mask = (speech_power - 2 * noise_power.reshape(-1, 1)) / speech_power
                mask = np.maximum(mask, 0.1)  # Apply floor to avoid extreme attenuation
                
                # Apply the mask and reconstruct the signal
                speech_stft_denoised = speech_stft * mask
                y = librosa.istft(speech_stft_denoised)
            
            if normalize:
                # Normalize audio to have consistent volume
                y = librosa.util.normalize(y)
            
            # Save the processed audio
            sf.write(output_file_path, y, sr)
            
            # Clean up temporary conversion file if created
            if input_file_path.endswith("temp_conversion.wav"):
                os.remove(input_file_path)
            
            return output_file_path
            
        except Exception as e:
            raise Exception(f"Audio preprocessing failed: {str(e)}")
    
    @staticmethod
    def convert_to_optimal_format(input_file_path, target_sr=16000):
        """
        Convert audio to an optimal format for Whisper (16kHz mono WAV).
        
        Args:
            input_file_path: Path to the input audio file
            target_sr: Target sample rate (Whisper works best with 16kHz)
            
        Returns:
            Path to the converted audio file
        """
        temp_dir = tempfile.mkdtemp()
        output_file_path = os.path.join(temp_dir, "whisper_optimized.wav")
        
        try:
            # Load audio
            y, sr = librosa.load(input_file_path, sr=None, mono=True)
            
            # Resample if needed
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            
            # Save as 16-bit PCM WAV (optimal for Whisper)
            sf.write(output_file_path, y, target_sr, subtype='PCM_16')
            
            return output_file_path
            
        except Exception as e:
            raise Exception(f"Audio format conversion failed: {str(e)}")