import json

with open('formatted_result_new.json', 'r') as f:
    data = json.load(f)

new_data = []
for i, item in enumerate(data[:3]):
    item['index'] = i
    item['id'] = int(item['images'][0].split('img_sample')[1].split('.')[0])
    item['messages'] = item['messages'][:2]  # Keep only the relevant messages
    item['images'][0] = item['images'][0].replace('/mnt/nas2/jeochris/resized_imgs', '/data/nas-1/images_10000_resized')
    new_data.append(item)
    with open('formatted_result_nas_path.json', 'w') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)