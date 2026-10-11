import os
import time
import subprocess
import aiohttp
from mistralai import Mistral
from config import MUNSIT_API_KEY, MISTRAL_API_KEY, MAX_STT_MINUTES

# Keep a simple global to track usage (in a real app, use redis/db)
USED_STT_MINUTES = 0.0

async def convert_audio_to_pcm16_16k(input_path, output_path):
    # Converts audio to 16kHz mono PCM16 wav
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        output_path
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise Exception(f"FFmpeg conversion failed: {stderr.decode()}")
        
def calculate_wer(reference, hypothesis):
    # Simplistic WER for demo purposes. Rapidfuzz could also be used.
    # Normalise Arabic is requested: remove diacritics, unify alef/yaa/taa marbuta
    import re
    def normalize(text):
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text) # diacritics
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'ى', 'ي', text)
        # remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower().strip()
        
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()
    
    # Very basic WER based on edit distance of words
    from rapidfuzz.distance import Levenshtein
    distance = Levenshtein.distance(ref_words, hyp_words)
    return distance / len(ref_words) if ref_words else 0.0

class STTProvider:
    async def transcribe(self, file_path, language, reference_text=None):
        global USED_STT_MINUTES
        start = time.perf_counter()
        
        # Audio length (approx via file size: 16k * 2 bytes = 32kB/s)
        file_size = os.path.getsize(file_path)
        duration_mins = (file_size / 32000) / 60
        
        if USED_STT_MINUTES + duration_mins > MAX_STT_MINUTES:
            return {"error": f"MAX_STT_MINUTES ({MAX_STT_MINUTES}) exceeded."}
            
        USED_STT_MINUTES += duration_mins
        
        result = {
            "transcript": "",
            "latency": 0,
            "speaker_labels": False,
            "error": None,
            "wer": None,
            "provider": ""
        }
        
        try:
            if language == "Arabic / mixed":
                result["provider"] = "munsit"
                async with aiohttp.ClientSession() as session:
                    with open(file_path, "rb") as f:
                        data = aiohttp.FormData()
                        data.add_field('file', f, filename="audio.wav", content_type="audio/wav")
                        data.add_field('model', 'munsit-en-ar')
                        
                        headers = {"x-api-key": MUNSIT_API_KEY}
                        
                        async with session.post("https://api.munsit.com/api/v1/audio/transcribe", data=data, headers=headers) as resp:
                            if resp.status == 200:
                                json_resp = await resp.json()
                                result["transcript"] = json_resp.get("data", {}).get("transcription", "")
                                # Check if turns are returned
                                if "turns" in json_resp.get("data", {}):
                                    result["speaker_labels"] = True
                            else:
                                err_text = await resp.text()
                                result["error"] = f"Munsit HTTP {resp.status}: {err_text}"
                                
            else: # English
                result["provider"] = "mistral"
                # For mistral we use batch since streaming isn't fully confirmed via SDK.
                # Standard transcription API
                mistral_client = Mistral(api_key=MISTRAL_API_KEY)
                
                # Using run_in_executor to avoid blocking since mistral sdk might be sync
                # Note: mistralai provides async but syntax varies, so we will use sync in a thread
                def _run_mistral():
                    with open(file_path, "rb") as f:
                        return mistral_client.audio.transcriptions.create(
                            model="mistral-small-2402", # Or whatever standard batch model
                            file=f
                        )
                        
                import asyncio
                loop = asyncio.get_event_loop()
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as pool:
                    resp = await loop.run_in_executor(pool, _run_mistral)
                
                result["transcript"] = resp.text
                
        except Exception as e:
            result["error"] = str(e)
            
        result["latency"] = time.perf_counter() - start
        
        if reference_text and not result["error"]:
            result["wer"] = calculate_wer(reference_text, result["transcript"])
            
        return result

import asyncio
