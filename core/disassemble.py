import os
import argparse
import json
import time
import ast
import torch
from typing import List
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import logging

from utils.utils import extract_activity, extract_date, extract_function_name, extract_time_of_day, full_time_of_day

def load_api_key(file_path='api_key.txt'):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('api_key='):
                return line.strip().split('=', 1)[1]
    return None

# Ensure the OpenAI API key is set
os.environ["OPENAI_API_KEY"] = load_api_key()

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

def get_function_names(filename):
    with open(filename, "r") as file:
        file_content = file.read()

    # Parse the content into an AST
    tree = ast.parse(file_content)

    # Traverse the AST to find function definitions
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    return function_names


# You can only select from candidate activities of {act_list}.
INSTRUCTION = \
"""You are a helpful assistant in searching and querying relevant sensor data from a database to answer the users' questions. The questions are about daily life activites.
What I want you to do is to generate a query to extract relevant sensor data via calling specific code functions. Please highlight the code function name with << >>, highlight the corresponding activity with (( )), highlight the corresponding date with [[ ]]. 
You can only call one function from {func_list}.
Below are some demonstration examples.\n"""

INSTRUCTION_NO_EXAMPLE = \
"""You are a helpful assistant in searching and querying relevant sensor data from a database to answer the users' questions. The questions are about daily life activites.
What I want you to do is to generate a query to extract relevant sensor data via calling specific code functions. Please highlight the code function name with << >>, highlight the corresponding activity with (( )), highlight the corresponding date with [[ ]]. 
You can only call one function from {func_list}.\n"""

EXAMPLE = \
"""
Question: {question}
Solution: {solution}
Explanation: {explanation}
"""

EXAMPLE_NO_COT = \
"""
Question: {question}
Solution: {solution}
"""

PROMPT = \
"""Now please generate the solution and step-by-step explanation for the following question.
Question: {question}
New solution:
"""

PROMPT_NO_COT = \
"""Now please generate the solution for the following question.
Question: {question}
New solution:
"""

def gpt_disassemble(question: str,
                    func_list: List[str],
                    act_list: List[str],
                    tmpl_filename: str,
                    llm_model: str='gpt-3.5-turbo',
                    no_example: bool=False,
                    no_cot: bool=False):
    """
    Query GPT to obtain the solution for the question
    """
    # put together the prompts to GPT
    func_list_str = ', '.join(func_list)
    act_list_str = ', '.join(act_list + ['all activities'])
    time_of_day_list_str = ', '.join(full_time_of_day)
    
    if not no_example: # with examples
        # put together the demo examples
        tmpl = json.load(open(tmpl_filename))
        examples = ""
        for exp in tmpl:
            if not no_cot: # with chain of thought
                examples += EXAMPLE.format(question=exp["Question"],
                                        solution=exp["Solution"],
                                        explanation=exp["Explanation"])
            else: # no chain of thought
                examples += EXAMPLE_NO_COT.format(question=exp["Question"],
                                        solution=exp["Solution"])
        system_prompt = INSTRUCTION.format(func_list=func_list_str, 
                                       act_list=act_list_str,
                                       time_of_day_list=time_of_day_list_str) + examples
    
    else:  # no example
        system_prompt = INSTRUCTION_NO_EXAMPLE.format(func_list=func_list_str, 
                                       act_list=act_list_str,
                                       time_of_day_list=time_of_day_list_str)

    if not no_cot: # with chain of thought
        user_prompt = PROMPT.format(question=question)
    else: # no chain of thought
        user_prompt = PROMPT_NO_COT.format(question=question)
    #print('system prompt: ', system_prompt)
    #print('user prompt: ', user_prompt)

    message = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    def make_openai_call():
        return client.chat.completions.create(
            model=llm_model,
            messages=message,
            max_tokens=256,
            temperature=1,
            top_p=1
        )

    try:
        response = make_openai_call()
    except:
        time.sleep(20)
        response = make_openai_call()

    response = json.loads(response.model_dump_json())
    r = response["choices"][0]["message"]["content"]
    #response_list = r.split("\n")
    #print(r)
    return r


def llama_disassemble(question: str,
                      func_list: List[str],
                      act_list: List[str],
                      tmpl_filename: str,
                      llm_model: str="llama2",
                      no_example: bool=False,
                      no_cot: bool=False):
    """
    Query llama to obtain the solution for the question
    """
    if not hasattr(llama_disassemble, "has_been_called"):
        llama_disassemble.has_been_called = True
        access_token = os.environ.get("HUGGINGFACE_API_KEY", None)

        if llm_model == "llama2":
            model = "meta-llama/Llama-2-7b-hf"
        elif llm_model == "llama3":
            model = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            raise ValueError(f"llm model {llm_model} is not implemented!")

        llama_disassemble.tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)

        compute_dtype = getattr(torch, "float16")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        llama_disassemble.model = AutoModelForCausalLM.from_pretrained(model, token=access_token, device_map='auto', quantization_config=quant_config)

    
    # put together the prompts to GPT
    func_list_str = ', '.join(func_list)
    act_list_str = ', '.join(act_list + ['all activities'])
    time_of_day_list_str = ', '.join(full_time_of_day)
    
    if not no_example: # with examples
        # put together the demo examples
        tmpl = json.load(open(tmpl_filename))
        examples = ""
        for exp in tmpl:
            if not no_cot: # with chain of thought
                examples += EXAMPLE.format(question=exp["Question"],
                                        solution=exp["Solution"],
                                        explanation=exp["Explanation"])
            else: # no chain of thought
                examples += EXAMPLE_NO_COT.format(question=exp["Question"],
                                        solution=exp["Solution"])
        system_prompt = INSTRUCTION.format(func_list=func_list_str, 
                                       act_list=act_list_str,
                                       time_of_day_list=time_of_day_list_str) + examples
    
    else:  # no example
        system_prompt = INSTRUCTION_NO_EXAMPLE.format(func_list=func_list_str, 
                                       act_list=act_list_str,
                                       time_of_day_list=time_of_day_list_str)

    if not no_cot: # with chain of thought
        user_prompt = PROMPT.format(question=question)
    else: # no chain of thought
        user_prompt = PROMPT_NO_COT.format(question=question)
    #print('system prompt: ', system_prompt)
    #print('user prompt: ', user_prompt)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = llama_disassemble.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llama_disassemble.model.device)

    terminators = [
        llama_disassemble.tokenizer.eos_token_id,
        llama_disassemble.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    result = llama_disassemble.model.generate(input_ids,
                                        max_new_tokens=256,
                                        eos_token_id=terminators,
                                        do_sample=True,
                                        temperature=0.6,
                                        top_p=0.9,
                                        pad_token_id=llama_disassemble.tokenizer.eos_token_id
                                        )
    response = result[0][input_ids.shape[-1]:]
    answer = llama_disassemble.tokenizer.decode(response, skip_special_tokens=True)
    #if "Solution" in answer:
    #    answer = [a for a in answer.split('\n\n') if "Solution" in a][0]
    #if "solution" in answer:
    #    answer = [a for a in answer.split('\n\n') if "solution" in a][0]

    return answer


def parse_solution(solution):
    # set default returns
    function_name = "calculate_duration"
    activity_list = ["all activities"]
    date = "all days"
    time_of_day_list = ['']

    try:
        function_name = extract_function_name(solution)[0]
    except:
        logging.debug(f"ERROR! No valid function name")
        function_name = "calculate_duration"

    try:
        activity_res = []
        activity = extract_activity(solution)
        if len(activity) <= 0 or sum([len(a) for a in activity]) <= 0:
            raise ValueError
        for a in activity:
            if ',' in a: # split activity into a list if there are multiple items
                activity_res.extend([aa.strip().lower() for aa in a.split(',')])
            else:
                activity_res.append(a.strip().lower())
        activity_list = list(set(activity_res))
    except:
        logging.debug(f"ERROR! No valid activity")
        activity_list = ["all activities"]

    try:
        date = extract_date(solution)[0]
    except:
        logging.debug(f"ERROR! No valid date")
        date = "all days"
    
    try:
        time_res = []
        time = extract_time_of_day(solution)
        if len(time) <= 0 or sum([len(t) for t in time]) <= 0:
            raise ValueError
        for t in time:
            if ',' in t: # split time into a list if there are multiple items
                time_res.extend([tt.strip().lower() for tt in t.split(',')])
            else:
                time_res.append(t.strip().lower())
        time_of_day_list = list(set(time_res))
    except:
        logging.debug(f"ERROR! No valid time of day")
        time_of_day_list = ['']

    return function_name, activity_list, date, time_of_day_list


def clean_date_info_in_parse(date):
    if date == 'last week':
        return 'all days'
    if date == 'this week':
        return 'all days'
    return date


def question_decompose(question, func_list, act_list, tmpl_filename, args):
    if 'gpt' in args.llm_decompose:
        solution = gpt_disassemble(question, func_list, act_list, tmpl_filename,
                                   llm_model=args.llm_decompose, 
                                   no_example=args.no_example, no_cot=args.no_cot)
    elif 'llama' in args.llm_decompose:
        solution = llama_disassemble(question, func_list, act_list, tmpl_filename,
                                     llm_model=args.llm_decompose,
                                     no_example=args.no_example, no_cot=args.no_cot)
    else:
        raise ValueError(f"llm model {args.llm_model} is not implemented!")
    logging.debug(solution)

    # process the solution
    func_name, activity_list, date, time_of_day_list = parse_solution(solution)
    logging.debug('{} {} {} {}'.format(func_name, activity_list, date, time_of_day_list))
    return func_name, activity_list, date, time_of_day_list, solution


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--templates_folder', type=str, default="templates", help="templates folder")
    parser.add_argument('--func_module', type=str, default="core/func", help="name of module that has all functions")

    args = parser.parse_args()

    # get all available functions
    func_names = get_function_names(args.func_module + '.py')
    func_list = ', '.join(func_names)

    question = 'How much time did I do groom on Wednesday'
    q_cat = 'Time Query'

    tmpl_name = '_'.join(q_cat.lower().split()) + '.json'
    tmpl_filename = os.path.join(args.templates_folder, tmpl_name)
    solution = llama_disassemble(question, func_list, tmpl_filename,
                                 llm_model="llama3")

    print(solution, '\n')
    function_name = extract_function_name(solution)
    activity = extract_activity(solution)
    date = extract_date(solution)
    print("function_name: ", function_name)
    print("activity: ", activity)
    print("date: ", date)
