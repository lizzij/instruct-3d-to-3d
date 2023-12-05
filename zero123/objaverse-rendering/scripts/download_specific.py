import objaverse
from tqdm import tqdm
import multiprocessing
processes = multiprocessing.cpu_count()
print(objaverse.__version__)
import json

uids = ['fef8c7a37ebe46efba7efce99e39c3e0']

objects = objaverse.load_objects(
    uids=uids,
    download_processes=processes
)
print(objects)