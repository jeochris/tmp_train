from openai import OpenAI
import json
from datasets import load_from_disk

client = OpenAI(
    base_url="http://localhost:8880/v1",
    api_key="EMPTY",
)

with open("../dataset/balanced_subset_idx_test_300.json", "r") as f:
    balanced_subset_idx = json.load(f)
with open("../dataset/formatted_result_new.json", "r") as f:
    formatted_result = json.load(f)

cnt = 0
answer = []
cnt2 = 0
answer2 = []

for i in balanced_subset_idx[1:]:
    data = formatted_result[i]
    main_c = data['messages'][1]['content'][0]['text']
    sub_c = data['messages'][3]['content'][0]['text']
    img_path = data['images'][0]
    chat_response = client.chat.completions.create(
        model = "vision",
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the sub category?"},
                {"type": "image_url", "image_url": {"url": f"file://{img_path}" }},
            ],
        }],
    )
    result = chat_response.choices[0].message.content.strip()
    print(result)

    answer.append(result)
    if result == main_c:
        cnt += 1

    # chat_response = client.chat.completions.create(
    #     model = "vision",
    #     messages = [{
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "What is the sub category?"},
    #             {"type": "image_url", "image_url": {"url": f"file://{img_path}" }},
    #         ],
    #     }],
    # )
    # result = chat_response.choices[0].message.content.strip()
    # print(result)

    # answer2.append(result)
    # if result == sub_c:
    #     cnt2 += 1
    # raise

print(f"Main Category Accuracy: {cnt / len(balanced_subset_idx)}")
print(f"Sub Category Accuracy: {cnt2 / len(balanced_subset_idx)}")
with open("main_category_result.json", "w") as f:
    json.dump(answer, f, indent=4, ensure_ascii=False)
with open("sub_category_result.json", "w") as f:
    json.dump(answer2, f, indent=4, ensure_ascii=False)