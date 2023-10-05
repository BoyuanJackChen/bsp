python scripts/benchmark.py \
       --model facebook/opt-6.7b \
       --assist-model facebook/opt-125m \
       --tokenizer facebook/opt-125m \
       --len-out 128 \
       --speculate-step 4 \
       --batch-size 8 \
       --fp16 \
       --dataset alespalla/chatbot_instruction_prompts \
       --dataset-truncate 200 \
       # --collect-stats