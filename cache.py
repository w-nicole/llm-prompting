from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import time

from tqdm import tqdm


def is_response_cached(id: str) -> bool:
    return os.path.isfile(f"cache/{id}.json")


def get_cached_response(id: str):
    return json.load(open(f"cache/{id}.json", "r"))


def cache_response(id: str, response):
    json.dump(response, open(f"cache/{id}.json", "w+"))


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


# For testing
# multithread_with_tqdm(download_file, 10, range(10), 3)
# print(is_response_cached("1"))
# print(cache_response("1", [1, 2, 3]))
# print(is_response_cached("1"))
# print(get_cached_response("1"))


# To replace original code
