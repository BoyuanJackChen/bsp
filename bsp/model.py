from typing import List
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import argparse

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

# def measure_time(, )

@torch.inference_mode()
def generate_hf(prompts, model, tokenizer, step):
    gen_conf = GenerationConfig(max_new_tokens=step, min_new_tokens=step, eos_token_id=tokenizer.eos_token_id)
    token_seqs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    token_seqs = token_seqs.to('cuda')
    model = model.to('cuda')
    out = model.generate(**token_seqs, generation_config=gen_conf)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--assist-model', type=str)
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--num-out', type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).half().cuda()
    assist_model = AutoModelForCausalLM.from_pretrained(args.assist_model).half().cuda()

    import time
    prompts = ["The University of Toronto is", "The city of Toronto is", "Canada is"]
    generator = SpeculativeGenerationModel(model, assist_model, tokenizer)
    # warm up
    generator.generate(prompts, args.step, args.num_out)

    torch.cuda.synchronize()
    start_t = time.time()
    spec_out = generator.generate(prompts, args.step, args.num_out)
    torch.cuda.synchronize()
    print("speculative:", time.time() - start_t)

    tokenizer.padding_side='left'
    generate_hf(prompts, model, tokenizer, args.num_out)
    torch.cuda.synchronize()
    start_t = time.time()
    ground_truth = generate_hf(prompts, model, tokenizer, args.num_out)
    torch.cuda.synchronize()
    print("baseline:", time.time() - start_t)

    for a, b in zip(spec_out, ground_truth):
        print('='*10)
        print(a)
        print('-'*10)
        print(b)
    assert(spec_out == ground_truth)
