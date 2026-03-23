import requests
import time
import statistics
import psutil
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:8000/embed"
TEST_TEXT = "Это пример текста средней длины для замера производительности модели на русском языке."

N_REQUESTS = 1000
WARMUP_REQUESTS = 30
N_THREADS = 50

print(f"Разогрев ({WARMUP_REQUESTS} запросов)...")
for _ in range(WARMUP_REQUESTS):
    requests.post(URL, json={"texts": [TEST_TEXT]})

def make_request():
    start = time.perf_counter()
    try:
        response = requests.post(URL, json={"texts": [TEST_TEXT]})
        if response.status_code == 200:
            latency = (time.perf_counter() - start) * 1000
            return latency
        else:
            return None
    except Exception:
        return None

print(f"Запуск теста: {N_REQUESTS} запросов, {N_THREADS} потоков...")

latencies = []
start_total = time.perf_counter()

process = psutil.Process()
mem_start = process.memory_info().rss / (1024 ** 2)

with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    futures = [executor.submit(make_request) for _ in range(N_REQUESTS)]

    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            latencies.append(result)

total_time = time.perf_counter() - start_total

mem_end = process.memory_info().rss / (1024 ** 2)

if latencies:
    stats = {
        "успешные_запросы": len(latencies),
        "всего_запросов": N_REQUESTS,
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_median_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(np.percentile(latencies, 95), 2),
        "latency_p99_ms": round(np.percentile(latencies, 99), 2),
        "throughput_req_per_sec": round(len(latencies) / total_time, 2),
        "общее_время_сек": round(total_time, 2),
        "память_до_МБ": round(mem_start, 1),
        "память_после_МБ": round(mem_end, 1),
        "память_дельта_МБ": round(mem_end - mem_start, 1)
    }

    print("\nРезультаты:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
else:
    print("Нет успешных запросов")