from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import time

from tqdm import tqdm

def is_response_cached(id: str, folder: str) -> bool:
    return os.path.isfile(f"{folder}/{id}.json")

def get_cached_response(id: str, folder: str):
    return json.load(open(f"{folder}/{id}.json", "r"))

def cache_response(id: str, folder: str, response):
    json.dump(response, open(f"{folder}/{id}.json", "w+"))

def multithread_with_tqdm(worker_fn, total, iterator, num_workers=10):
    with tqdm(total=total) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(worker_fn, item, pbar) for item in iterator]
            for future in as_completed(futures):
                future.result()

def download_file(url, pbar):
    time.sleep(0.50 * random.random())
    print(url)
    pbar.update(1)