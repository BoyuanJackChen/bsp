python scripts/benchmark.py \
       --model facebook/opt-6.7b  \
       --assist-model facebook/opt-125m\
       --tokenizer facebook/opt-125m\
       --len-out 128 \
       --speculate-step 0 1 2 3 4 5 6 7 8\
       --batch-size 1 2 4 8 16 32\
       --fp16 \
       --dataset alespalla/chatbot_instruction_prompts \
       --dataset-truncate 3 \
       # --collect-stats

python scripts/benchmark.py \
       --model facebook/opt-6.7b  \
       --assist-model facebook/opt-125m\
       --tokenizer facebook/opt-125m\
       --len-out 128 \
       --speculate-step 4\
       --batch-size 4\
       --fp16 \
       --dataset alespalla/chatbot_instruction_prompts \
       --dataset-truncate 8 
       # --collect-stats

python scripts/benchmark.py \
       --model WizardLM/WizardCoder-3B-V1.0 \
       --assist-model WizardLM/WizardCoder-1B-V1.0\
       --tokenizer WizardLM/WizardCoder-1B-V1.0\
       --len-out 512 \
       --speculate-step 4\
       --batch-size 2\
       --fp16 \
       --dataset humaneval \
       --dataset-truncate 2
       # --collect-stats