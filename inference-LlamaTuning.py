import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import AutoPeftModelForCausalLM


BASE_MODEL_PATH = "/pretraining/pre-model-full-multi/final"  
ADAPTER_PATH = "/fine_tuning/final_adapter"
QUESTION_FILE = "question.txt"
ANSWER_FILE = "answer.txt"

target_gpu_id = 5


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)


model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_PATH,
    torch_dtype=torch.bfloat16,
    device_map=target_gpu_id,
    trust_remote_code=False,
)


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map=target_gpu_id,
)


questions = []
with open(QUESTION_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            questions.append(line)


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


with open(ANSWER_FILE, "w", encoding="utf-8") as out_file:
    for i, question in enumerate(questions, 1):


        messages = [
            {"role": "user", "content": question}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,          
            add_generation_prompt=True  
        )


        outputs = pipe(
            prompt,
            max_new_tokens=450,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.3,
            top_p=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id, 
        )


        generated_text = outputs[0]["generated_text"][len(prompt):]


        generated_text = generated_text.replace("<|eot_id|>", "").strip()


        out_file.write(f"Input: {question}\nOutput: {generated_text}\n\n")
        print(f"✅ Output: {generated_text}")

