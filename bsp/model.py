from typing import List
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import argparse
import time
import random

class SpeculativeGenerationModel:
    def __init__(self, model, assist_model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.assist_model = assist_model.to(device)
        self.tokenizer = tokenizer
        self.device=device

    def _speculative(self, input_ids, attention_mask, kv_cache, steps):
        batch_size = input_ids.shape[0]
        generated_tokens = [[] for _ in range(batch_size)]
        for i in range(steps):
            ret = self.assist_model(input_ids,
                                    attention_mask=attention_mask, 
                                    use_cache=True, 
                                    past_key_values=kv_cache)
            input_ids = torch.argmax(ret.logits[:, -1:], axis=2)

            for b in range(batch_size):
                generated_tokens[b].append(input_ids[b, 0])

            attention_mask = self._extend_mask(attention_mask) 
            kv_cache = ret.past_key_values
        return generated_tokens, attention_mask, kv_cache
    
    def _last_pos_logits(self, logits, mask):
        last_pos = torch.sum(mask, axis=1) - 1
        return logits[torch.arange(logits.shape[0]), last_pos]
    
    def _extend_mask(self, mask):
        return torch.cat([mask, torch.ones([mask.shape[0], 1], device=self.device, dtype=torch.int32)], axis=1)

    @torch.inference_mode()
    def generate(self, prompts:List[str], specualtive_step:int, num_out:int):
        tokenizer.padding_side='right'
        token_seqs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        batch_size = len(prompts)
        assist_kv_cache = None
        input_ids = token_seqs['input_ids'].to(self.device)
        attention_mask = input_attention_mask = token_seqs['attention_mask'].to(self.device)
        prompt_len = attention_mask.sum(axis=1)

        # prefill
        ret = self.model(input_ids, attention_mask=input_attention_mask, use_cache=True)
        first_token = torch.argmax(self._last_pos_logits(ret.logits, attention_mask), axis=1).unsqueeze(1) 
        attention_mask = self._extend_mask(attention_mask)
        input_ids = torch.cat([input_ids, first_token], axis=1)
        kv_cache = ret.past_key_values
        generated_tokens = input_ids
        valid_lens = torch.ones(batch_size, device=self.device) 

        while True:
            speculated_tokens, attention_mask, assist_kv_cache = self._speculative(input_ids, attention_mask, assist_kv_cache, specualtive_step)
            # verify
            speculated_tokens = torch.tensor(speculated_tokens, device=self.device)
            verify_inputs = torch.cat([first_token, speculated_tokens], axis=1)
            ret = self.model(verify_inputs, attention_mask=attention_mask, use_cache=True, past_key_values=kv_cache)
            logits = ret.logits
            kv_cache = ret.past_key_values
            correct = logits[:, :-1].argmax(dim=2)

            # mask wrong predictions
            check_mask = torch.cumsum(correct == speculated_tokens, 1) == torch.arange(1, specualtive_step + 1, device=self.device)
            correct_len = torch.sum(check_mask, axis=1)
            first_token = torch.argmax(logits[torch.arange(logits.shape[0]), correct_len], axis=1).unsqueeze(1)
            input_ids = torch.concat([speculated_tokens[:, -1:], first_token], axis=1)
            attention_mask[:, -specualtive_step:] = check_mask
            attention_mask = self._extend_mask(attention_mask)
            generated_tokens = torch.cat([generated_tokens, speculated_tokens, first_token], axis=1)
            valid_lens += correct_len + 1
            if torch.all(valid_lens >= num_out):
                break
        ret = []
        for b in range(batch_size):
            valid_token = torch.nonzero(attention_mask[b], as_tuple=True)[0]
            tokens = generated_tokens[b][valid_token][:prompt_len[b] + num_out]
            ret.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        return ret

def benchmark(fn, num=1, warmup=3, ground_truth=None):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start_t = time.time()
    for _ in range(num):
        out = fn()
    torch.cuda.synchronize()
    dur = (time.time() - start_t) / num
    if ground_truth is not None:
        assert out == ground_truth
    return dur

@torch.inference_mode()
def generate_hf(prompts, model, tokenizer, step):
    tokenizer.padding_side='left'
    gen_conf = GenerationConfig(max_new_tokens=step, min_new_tokens=step, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    token_seqs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    token_seqs = token_seqs.to('cuda')
    model = model.to('cuda')
    out = model.generate(**token_seqs, generation_config=gen_conf)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

@torch.inference_mode()
def generate_hf_assist(prompts, model, assist_model, tokenizer, step):
    tokenizer.padding_side='left'
    gen_conf = GenerationConfig(max_new_tokens=step, min_new_tokens=step, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    ret = []
    for p in prompts:
        token_seqs = tokenizer(p, return_tensors="pt")
        # print(token_seqs)
        token_seqs = token_seqs.to('cuda')
        model = model.to('cuda')
        out = model.generate(**token_seqs, generation_config=gen_conf, assistant_model=assist_model)
        ret.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--assist-model', type=str)
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--num-out', type=int)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--print', action='store_true')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).half().cuda()
    assist_model = AutoModelForCausalLM.from_pretrained(args.assist_model).half().cuda()
    assist_model.max_assistant_tokens = args.step

    prompts = random.choices(["The University of Toronto is", "The city of Toronto is", "Canada is"], k=args.batch_size)
    generator = SpeculativeGenerationModel(model, assist_model, tokenizer)
    ground_truth = generate_hf(prompts, model, tokenizer, args.num_out)
    output = generator.generate(prompts, args.step, args.num_out)
    # print(ground_truth)
    print("baseline:", benchmark(lambda: generate_hf(prompts, model, tokenizer, args.num_out)))
    # print("speculative:", benchmark(lambda: generate_hf_assist(prompts, model, assist_model, tokenizer, args.num_out)))
    print("batched speculative:", benchmark(lambda : generator.generate(prompts, args.step, args.num_out), ground_truth=ground_truth))
    
    if args.print:
        for a, b in zip(ground_truth, output):
            print('='*10)
            print(a)
            print('-'*10)
            print(b)