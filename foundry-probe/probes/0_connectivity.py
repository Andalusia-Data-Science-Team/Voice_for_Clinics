import asyncio
import socket
import ssl
import time
import urllib.parse
from statistics import median
import aiohttp
from config import client, FOUNDRY_BASE_URL, FOUNDRY_PROJECT_ENDPOINT, DEPLOYMENTS

async def measure_network(url, samples=10):
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    
    print(f"--- Network Connectivity to {host}:{port} ---")
    
    # DNS
    try:
        ip = socket.gethostbyname(host)
        print(f"DNS Resolution: OK ({ip})")
    except Exception as e:
        print(f"DNS Resolution: FAILED ({e})")
        return

    ttfb_times = []
    for i in range(samples):
        try:
            start_time = time.perf_counter()
            # TCP
            s = socket.create_connection((host, port), timeout=5)
            # TLS
            if parsed.scheme == 'https':
                context = ssl.create_default_context()
                s = context.wrap_socket(s, server_hostname=host)
            
            # Send dummy HTTP request for TTFB
            request = f"GET {parsed.path or '/'} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
            s.sendall(request.encode())
            
            # Wait for first byte
            s.recv(1)
            end_time = time.perf_counter()
            ttfb_times.append((end_time - start_time) * 1000)
            s.close()
        except Exception as e:
            print(f"Sample {i+1} FAILED: {e}")
    
    if ttfb_times:
        print(f"TCP/TLS + TTFB (10 samples): Median {median(ttfb_times):.1f}ms, Max {max(ttfb_times):.1f}ms")
    else:
        print("All network samples failed.")

async def test_openai_api():
    print("\n--- OpenAI API Connectivity ---")
    try:
        models = await client.models.list()
        print(f"models.list(): OK (Found {len(models.data)} deployments)")
    except Exception as e:
        print(f"models.list(): FAILED ({e.__class__.__name__}: {e})")
        print("Falling back to probing individual deployments...")
        
        for deployment in DEPLOYMENTS:
            try:
                await client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1
                )
                print(f"Deployment '{deployment}': OK")
            except Exception as de:
                err_str = str(de)
                if "DeploymentNotFound" in err_str or "404" in err_str:
                    print(f"Deployment '{deployment}': NOT FOUND (404)")
                elif "401" in err_str or "auth" in err_str.lower():
                    print(f"Deployment '{deployment}': AUTH ERROR (401)")
                else:
                    print(f"Deployment '{deployment}': ERROR ({err_str})")

async def test_project_endpoint():
    print("\n--- Foundry Project Endpoint ---")
    if not FOUNDRY_PROJECT_ENDPOINT:
        print("No FOUNDRY_PROJECT_ENDPOINT provided.")
        return
        
    try:
        async with aiohttp.ClientSession() as session:
            # Try a basic GET request (listing)
            async with session.get(FOUNDRY_PROJECT_ENDPOINT, timeout=10) as resp:
                if resp.status == 401 or resp.status == 403:
                    print(f"Endpoint requires Entra ID / Auth (Status {resp.status}). Skipping.")
                elif resp.status == 200:
                    print("Endpoint accessed successfully without additional auth.")
                else:
                    print(f"Endpoint returned status {resp.status}.")
    except Exception as e:
        print(f"Project endpoint test failed: {e}")

async def run():
    await measure_network(FOUNDRY_BASE_URL)
    await test_openai_api()
    await test_project_endpoint()

if __name__ == "__main__":
    asyncio.run(run())
