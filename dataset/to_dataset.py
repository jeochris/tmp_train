from datasets import Dataset, Features, Sequence, Value, Image, load_dataset, DatasetDict
from pathlib import Path
import json

# 0️⃣ JSON 파일 저장 (이미 되어 있을 것)
json_path = "formatted_result_new.json"
dataset = Dataset.from_json(json_path).cast_column("images", Sequence(Image()))
print(dataset[0])

test_idx = json.load(open('balanced_subset_idx_test_300.json'))
train_idx = [i for i in range(5000) if i not in test_idx]
test_dataset = dataset.select(test_idx)
train_dataset = dataset.select(train_idx)
print(len(train_dataset), len(test_dataset))

DatasetDict({"train": train_dataset, "test": test_dataset}).save_to_disk("/mnt/nas2/jeochris/yanolja_info_dataset")