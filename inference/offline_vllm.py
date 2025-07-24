from vllm import LLM, SamplingParams, EngineArgs
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import MediaConnector
from transformers import AutoProcessor, AutoTokenizer

from PIL.Image import Image
from typing import NamedTuple, Optional
from dataclasses import asdict

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None

def load_qwen2_5_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import smart_resize
    except ModuleNotFoundError:
        print(
            "WARNING: `qwen-vl-utils` not installed, input images will not "
            "be automatically resized. You can enable this functionality by "
            "`pip install qwen-vl-utils`."
        )
        smart_resize = None

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Tested on L40
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768 if smart_resize is None else 4096,
        enable_lora=True,  # 반드시 지정
        tensor_parallel_size=1,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
        allowed_local_media_path="/home/jeochris/train",
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    media_connector = MediaConnector(
        allowed_local_media_path=engine_args.allowed_local_media_path
    )

    if smart_resize is None:
        image_data = [media_connector.fetch_image(url) for url in image_urls]
    else:

        def post_process_image(image: Image) -> Image:
            width, height = image.size
            resized_height, resized_width = smart_resize(
                height, width, max_pixels=1024 * 28 * 28
            )
            return image.resize((resized_width, resized_height))

        image_data = [post_process_image(media_connector.fetch_image(url)) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
        lora_requests=LoRARequest("vision", 1, "/home/jeochris/train/test/checkpoint-507")
    )

req_data = load_qwen2_5_vl(
    question="What is the main category of facility?",
    image_urls=["file:///home/jeochris/train/resized_imgs/img_sample1.jpg"]
)
engine_args = asdict(req_data.engine_args) | {"seed": 0}
llm = LLM(**engine_args)

sampler = SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=req_data.stop_token_ids)

out = llm.chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the main category of facility?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "file:///home/jeochris/train/resized_imgs/img_sample1.jpg"}
                    }
                ]
            }
        ],
        sampling_params=sampler,
        chat_template=req_data.chat_template,
        lora_request=req_data.lora_requests,
)
print(out[0].outputs[0].text)
