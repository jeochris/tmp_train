import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from peft import LoraConfig, TaskType

parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
script_args, training_args, model_args = parser.parse_args_and_config()
training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
training_args.remove_unused_columns = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}

################
# Model, Tokenizer & Processor
################
torch_dtype = (
    model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
)
quantization_config = get_quantization_config(model_args)
model_kwargs = dict(
    revision=model_args.model_revision,
    attn_implementation=model_args.attn_implementation,
    torch_dtype=torch_dtype,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)
processor = AutoProcessor.from_pretrained(
    model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
)

model = AutoModelForVision2Seq.from_pretrained(
    model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
)

def collate_fn(examples):
    # print(examples)
    # raise
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["images"] for example in examples]
    if isinstance(model, LlavaForConditionalGeneration):
        # LLava1.5 does not support multiple images
        images = [image[0] for image in images]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

image_path = 'lose.png'
ex = [
    {
        "messages": [
            {
                "content": [
                    {
                        "index": None, "text": "Describe an image", "type": "text"
                    },
                    {
                        "index": 0, "text": None, "type": "image"
                    }
                ], 
                "role": "user"
            },
            {
                "content": [
                    {
                        "index": None, "text": "This is an image of a person losing a game.", "type": "text"
                    }
                ],
                "role": "assistant"
            }
        ],
        "images": [Image.open(image_path)]
    }
]

print(collate_fn(ex))