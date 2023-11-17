import objaverse
from tqdm import tqdm
import multiprocessing
processes = multiprocessing.cpu_count()
print(objaverse.__version__)
import json

uids = objaverse.load_uids()
len(uids), type(uids)

objects = objaverse.load_objects(
    uids=uids[:100],
    download_processes=processes
)

with open("input_models_path.json", "w") as f:
    paths = list(objects.values())
    json.dump(objects, f, indent=2)