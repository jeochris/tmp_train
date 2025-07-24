vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --dtype bfloat16 \
    --enable-lora \
    --lora-modules vision=/home/jeochris/yanolja/test/checkpoint-294 \
    --task generate \
    --allowed-local-media-path /mnt/nas2/jeochris/resized_imgs \
    --limit-mm-per-prompt image=1 \
    --port 8880 \