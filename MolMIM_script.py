import json
import math
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

url = "http://localhost:8000/generate"

TARGET_MOLECULES = 100000
BATCH_SIZE = 100
N_BATCHES = math.ceil(TARGET_MOLECULES / BATCH_SIZE)

base_payload = {
    "algorithm": "CMA-ES",
    "num_molecules": BATCH_SIZE,
    "property_name": "QED",
    "minimize": False,
    "min_similarity": 0.3,
    "particles": BATCH_SIZE,   # must be >= num_molecules
    "iterations": 3,
    "smi": "C1=CC=C(C=C1)C(CC(=O)O)CN",
}

output_file = "third_100k_compounds.txt"
progress_file = "generation_progress.txt"

session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[502, 503, 504],
    allowed_methods=frozenset(["POST"]),
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

written_records = 0
start_time = time.time()

with open(output_file, "w", encoding="utf-8") as out, open(progress_file, "w", encoding="utf-8") as prog:
    for batch in range(N_BATCHES):
        try:
            r = session.post(url, json=base_payload, timeout=(10, 300))
            print(f"Batch {batch + 1}/{N_BATCHES} status: {r.status_code}")

            if not r.ok:
                print("Error body:", r.text)
                prog.write(f"FAILED at batch {batch + 1}\n")
                prog.write(r.text + "\n")
                prog.flush()
                break

            batch_data = r.json()

            if isinstance(batch_data, list):
                for item in batch_data:
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written_records += 1
            else:
                out.write(json.dumps(batch_data, ensure_ascii=False) + "\n")
                written_records += 1

            out.flush()

            elapsed = time.time() - start_time
            rate = (batch + 1) / elapsed if elapsed > 0 else 0
            remaining_batches = N_BATCHES - (batch + 1)
            eta_seconds = remaining_batches / rate if rate > 0 else 0

            prog.seek(0)
            prog.truncate()
            prog.write(f"Completed batch {batch + 1}/{N_BATCHES}\n")
            prog.write(f"Records written: {written_records}\n")
            prog.write(f"Elapsed seconds: {elapsed:.1f}\n")
            prog.write(f"Estimated remaining seconds: {eta_seconds:.1f}\n")
            prog.flush()

            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            print(f"Batch {batch + 1} failed: {e}")
            prog.write(f"EXCEPTION at batch {batch + 1}: {e}\n")
            prog.flush()
            break

print(f"Done. Saved {written_records} records to {output_file}")
