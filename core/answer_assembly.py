
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_function_inference(question, sensor_results):
    example = "<s>[INST] <<SYS>>Answer the question based on the context below. " + \
        f"\n\n### Context: {sensor_results} " + \
        f"\n\n### Question: {question} [/INST]" + \
        f"\n\n### Response:"
    return example


def filter_answer(answer):
    start = "[ANS]"
    end = "[/ANS]"
    pattern_start = re.escape(start) + "(?=(.*))"
    pattern_end = f"(?=(.){re.escape(end)})"
    match_start = re.search(pattern_start, answer).end() if start in answer else 0
    match_end = re.search(pattern_end, answer).start(1) if end in answer else None
    #print('matched: ', mgt[match_start:match_end].strip())
    answer = answer[match_start:match_end].strip()
    return answer

class llama_assembly:
    def __init__(self, base_model, lora_ckpt_path, from_awq=False, awq_path=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
        from transformers import AutoConfig, GenerationConfig, LlamaForCausalLM
        from peft import PeftModel
        from awq import AutoAWQForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        if from_awq:
            model = AutoAWQForCausalLM.from_quantized(awq_path)

            config = AutoConfig.from_pretrained(base_model)
            model.config = config

            generation_config = GenerationConfig.from_model_config(config)
            model.generation_config = generation_config

            model.can_generate = LlamaForCausalLM.can_generate
            model.device = device

        else:  # Load finetuned lora adapter
            model = PeftModel.from_pretrained(model, lora_ckpt_path)
            model = model.merge_and_unload()

        self.pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50, temperature=0.5)

    def generate(self, question, sensor_results):
        prompt = preprocess_function_inference(question, sensor_results)
        result = self.pipe(f"{prompt}")
        answer = result[0]['generated_text']
        answer = answer.split('[/INST]')[1].strip()
        return answer


class nanollm_assembly:
    def __init__(self, model, api='mlc', quantization='q4f16_ft'): # The default api is 'mlc' and the default quantization is 'q4f16_ft'
        from nano_llm import NanoLLM, ChatHistory

        self.model = NanoLLM.from_pretrained(model=model, api=api, quantization=quantization)

        # create the chat history
        self.chat_history = ChatHistory(self.model, chat_template='llama-2',
                                        system_prompt="You are a helpful and friendly AI assistant in answering questions relevent to sensor context.")
        

    def generate(self, question, sensor_results):
        prompt = preprocess_function_inference(question, sensor_results)
        # add user prompt and generate chat tokens/embeddings
        self.chat_history.append('user', prompt)
        embedding, position = self.chat_history.embed_chat(use_cache=False)

        # generate bot reply
        reply = self.model.generate(
            embedding, 
            streaming=True, 
            kv_cache=self.chat_history.kv_cache,
            stop_tokens=self.chat_history.template.stop,
            max_new_tokens=50,
            temperature=0.5
        )
        # stream and print the output
        for token in reply:
            print(token, end='\n\n' if reply.eos else '', flush=True)
        return None


if __name__ == '__main__':
    obj = llama_assembly('NousResearch/Llama-2-7b-hf', 'sensist_new')
