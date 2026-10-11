import asyncio
import json
import wave
import time
import websockets
from config import MUNSIT_API_KEY

async def stream_munsit(audio_path):
    print("--- Munsit Streaming STT Probe ---")
    
    if not MUNSIT_API_KEY or MUNSIT_API_KEY == "dummy":
        print("Skipping: No MUNSIT_API_KEY provided.")
        return
        
    try:
        wav = wave.open(audio_path)
        pcm = wav.readframes(wav.getnframes())
    except Exception as e:
        print(f"Failed to read audio file: {e}")
        return

    url = f"wss://api.munsit.com/api/v1/listen?api_key={MUNSIT_API_KEY}&encoding=linear16&sample_rate=16000&interim_results=true"
    
    start_time = time.perf_counter()
    first_partial_time = None
    final_text = ""
    
    try:
        async with websockets.connect(url) as ws:
            async def send_audio():
                chunk = 6400  # 200 ms of 16 kHz 16-bit mono
                for i in range(0, len(pcm), chunk):
                    await ws.send(pcm[i:i + chunk])
                    await asyncio.sleep(0.2)  # pace at real time
                await ws.send(json.dumps({"type": "CloseStream"}))

            sender = asyncio.create_task(send_audio())
            async for message in ws:
                event = json.loads(message)
                
                if event["type"] == "Results":
                    if first_partial_time is None:
                        first_partial_time = time.perf_counter() - start_time
                        print(f"Time to first partial: {first_partial_time:.2f}s")
                        
                    if event["is_final"]:
                        final_text += event["transcript"] + " "
                        print("FINAL:", event["transcript"])
                        
                elif event["type"] == "Error":
                    print("Error:", event.get("code"), event.get("message"))
                    
                elif event["type"] == "Metadata" and "audio_seconds_billed" in event:
                    print("Billed seconds:", event["audio_seconds_billed"])
                    break
                    
            await sender
            
    except Exception as e:
        print("WebSocket Error:", str(e))
        
    total_time = time.perf_counter() - start_time
    print(f"Total stream duration: {total_time:.2f}s")
    print(f"Full transcript: {final_text}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        asyncio.run(stream_munsit(sys.argv[1]))
    else:
        print("Provide path to 16kHz mono WAV file.")
