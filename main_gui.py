##############################################################################
# Demo file for running extrasensory data online
##############################################################################
import os
import numpy as np
import time
import argparse
import logging
import json
import torch
import joblib
import importlib
from transformers import BertTokenizer, BertForSequenceClassification
from core.disassemble import get_function_names, question_decompose, clean_date_info_in_parse
from core.answer_assembly import llama_assembly, nanollm_assembly, filter_answer
from utils.es_data_utils import get_label_pretty_name, merged_label_names
#from awq import AutoAWQForCausalLM
all_activities = list(map(get_label_pretty_name, merged_label_names))
from utils.es_data_utils import all_user_ids
from utils.utils import match_list, full_time_of_day, find_most_frequent_element

all_activities = list(map(get_label_pretty_name, merged_label_names))

from flask import Flask, render_template, jsonify, request

def read_sensor_data(record, args):
    # get user id for sensor data
    sensor_file_name = record["image_url"].split("/")
    # Need to consider two types of path hierarchies, w/ or w/o an intermediate folder named figures
    if sensor_file_name[-2] == 'figures':
        subset, image_name = int(sensor_file_name[-3].split("_")[-1]), sensor_file_name[-1].split('_', 1)[-1]
    else:
        subset, image_name = int(sensor_file_name[-2].split("_")[-1]), sensor_file_name[-1].split('_', 1)[-1]
    # Correct
    subset = 1 if subset == 130 else subset

    user_number = int(image_name[image_name.find("usr"):].split("_")[0][3:]) + (21 if 5 < subset < 10 else (41 if 10 <= subset else 0))
    user_id = all_user_ids[user_number]
    data = np.load(os.path.join(args.sensor_data_path, '{}.npz'.format(user_id)))

    label_subset = sensor_file_name[-1].split('_')[0]
    return data, user_number, label_subset


def sensor_query(sensor_data, func_name, activity_list, date, time_of_day_list,
                 all_dates, func_names, candidate_labels, args):

    fn_list, act_list, date_list, t_list = [], [], [], []

    if func_name in func_names:
        fn_list.append(func_name)
    
    for a in activity_list:
        if len(a) <= 0:
            continue
        matched_a, _ = match_list(a, candidate_labels + ['all activities'])
        act_list.append(matched_a)
    
    for t in time_of_day_list:
        t = '' if ' day' in t else t
        matched_t, _ = match_list(t, full_time_of_day + [''])
        t_list.append(matched_t)

    date_list.append(date)

    # get func name
    func_name = find_most_frequent_element(fn_list)
    module = importlib.import_module(args.func_module)
    func = getattr(module, func_name)

    # get activity, date and time of day
    activity_list = list(set(act_list))        
    date = find_most_frequent_element(date_list)
    time_of_day_list = list(set(t_list))

    print('Finalized: ', func_name, activity_list, date, time_of_day_list)

    #### Query sensor results
    # get the index for candidate_labels in all_activites
    ind = np.array([all_activities.index(a) for a in candidate_labels])

    # call the corresponding function to obtain sensor results
    sensor_results = ""
    for act in activity_list:
        for time_of_day in time_of_day_list:
            sensor_results += func(sensor_data['Y'][:, ind],
                                sensor_data['T'],
                                act,
                                candidate_labels,
                                date, 
                                all_dates,
                                time_of_day)
    
    logging.debug(sensor_results)
    return(sensor_results)


def print_latency(time_profile):
    for key in time_profile:
        print(f'Avg latency for {key}: ', sum(time_profile[key]) / len(time_profile[key]))


# Classify questions
def classify_question(question, tokenizer_q, model_q, label_encoder_q, device):
    with torch.no_grad():
        encoding = tokenizer_q.encode_plus(
            question,
            add_special_tokens=True,
            max_length=64,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model_q(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        return label_encoder_q.inverse_transform(preds.cpu().numpy())[0]
    
#Map labels and answers
def map_labels(num):
    label_mapping = {
        1: "Existence",
        2: "Counting",
        3: "Action Query",
        4: "Time Query",
        5: "Day Query",
        6: "Time Compare"
    }
    return label_mapping.get(num, np.nan)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logging level control example.')
    parser.add_argument('--log', dest='log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('--llm_decompose', type=str, default="gpt-3.5-turbo", 
                        choices=["gpt-3.5-turbo", "gpt-4", "llama2", "llama3"],
                        help="llm models in question decomposition")
    parser.add_argument('--sensor_data_path', type=str, default='mlc_embeddings_pred', help="the path to ExtraSensory.per_uuid_features_labels")
    parser.add_argument('--qa_file', type=str, default="sensorqa_dataset/overall_sensorqa_dataset.json", 
                            help="sensorqa dataset folder to process")
    parser.add_argument('--templates_folder', type=str, default="icl_templates", help="templates folder")
    parser.add_argument('--func_module', type=str, default="func", help="name of module that has all functions")
    parser.add_argument('--no_example', action='store_true', help="whether to disable solution templates") ## NOT USED, NEEDED AS PLACEHOLDER
    parser.add_argument('--no_cot', action='store_true', help="whether to disable chain of thought") ## NOT USED, NEEDED AS PLACEHOLDER

    # Image directory
    parser.add_argument('--pred_img_path', type=str, default='pred_graphs', help="the path to the predicted image path")
    parser.add_argument('--oracle_img_path', type=str, default='oracle_graphs', help="the path to the oracle image path")

    # Model loading and checkpoints
    parser.add_argument('--llm_answer', type=str, default='NousResearch/Llama-2-7b-hf', help="llm base models in answer assembly")
    parser.add_argument('--lora_ckpt_path', type=str, default='sensist_new', 
                        help="saved LoRA adapter checkpoint model to load on top of the base model")
    parser.add_argument('--from_awq', action='store_true', help="whether to load awq quantized model")
    parser.add_argument('--awq_path', type=str, default='sensorqa/sensist_new_awq', help="saved awq quantized model")
    parser.add_argument('--nano_llm', action='store_true', help="whether to use nanollm")
    parser.add_argument('--subject', type=int, help="subject number")
    args = parser.parse_args()

    return args

#####################################
# Globel setup
#####################################
args = parse_arguments()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
numeric_level = getattr(logging, args.log_level.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {args.log_level}')

logging.basicConfig(level=numeric_level,
                    format='%(asctime)s - %(levelname)s - %(message)s')

func_names = get_function_names(args.func_module + '.py')
#print(func_names)

subject_num = args.subject

# load qa dataset
with open(args.qa_file, "r") as json_file:
    qa_data = json.load(json_file)

# load meta data
folder = args.qa_file.split('/')[0]  # Extract the folder name from the path
all_dates = json.load(open(os.path.join(folder, 'full_dates.json')))

if not args.nano_llm:
    # Load llama assembly 
    llama_model = llama_assembly(args.llm_answer, args.lora_ckpt_path, 
                                from_awq=args.from_awq, awq_path=args.awq_path)
else:
    # Load the nanollm environment and the awq quantized model
    llama_model = nanollm_assembly(args.llm_answer)

# Load question classification model
# Load question model
label_encoder_q = joblib.load('q_model/q_label_encoder.pkl')
tokenizer_q = BertTokenizer.from_pretrained('q_model')
model_q = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_q.classes_))
model_q.load_state_dict(torch.load('q_model/question_model.pt'))
model_q.to(device)
model_q.eval()

log = []

# Find one image example for QA demo
for rid, record in enumerate(qa_data):

    #### Read data
    #question = record['question']
    #answer = record['answer']
    #q_cat = record['pred_q_cat']
    #a_cat = record['pred_a_cat']
    today = record['today']
    #candidate_labels = record['candidate_labels']

    # Filter the single-day query
    if len(today) > 0:
        continue

    sensor_data, user_number, label_subset = read_sensor_data(record, args)

    if user_number == args.subject: # This is the data we find!
        path_to_figure = os.path.join(args.oracle_img_path, f'{label_subset}_usr{user_number}_weekly.png')
        print('Sensor graph path: ', path_to_figure)
        break

time_profile = {
    'all': [],
    'question_decompose': [],
    'sensor_query': [],
    'answer_assembly': []
}

#####################################
# Key demo
#####################################
app = Flask(__name__)

# Serve the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get bot response
@app.route('/get-bot-response', methods=['POST'])
def get_bot_response():
    data = request.get_json()
    question = data.get('user_message', '')

    # Start the demo loop
    new_log = {}

    #############################
    #### Decompose questions
    #############################
    decompose_start = time.time()
    # Classify question type
    q_cat = map_labels(int(classify_question(question, tokenizer_q, model_q, label_encoder_q, device)))
    print('question type: ', q_cat)
    tmpl_name = '_'.join(q_cat.lower().split()) + '.json'
    tmpl_filename = os.path.join(args.templates_folder, tmpl_name)
    #tmpl_filename = os.path.join(args.templates_folder, 'all.json')

    # decompose the question!
    func_name, activity_list, date, time_of_day_list, solution = \
        question_decompose(question, func_names, all_activities, tmpl_filename, args)
    date = clean_date_info_in_parse(date)

    time_profile['question_decompose'].append(time.time() - decompose_start)

    ###################################################
    #### Sensor query
    ###################################################
    query_start = time.time()
    sensor_results = sensor_query(sensor_data, func_name, activity_list, date, time_of_day_list,all_dates[str(args.subject)], func_names, all_activities, args)
    print('sensor results: ', sensor_results)
    time_profile['sensor_query'].append(time.time() - query_start)

    ###############################
    ##### Answer assembly
    ###############################
    assembly_start = time.time()
    answer = llama_model.generate(question, sensor_results)
    end = time.time()

    time_profile['answer_assembly'].append(end - assembly_start)
    time_profile['all'].append(end - decompose_start)
    
    if answer is not None: # non-nanollm outputs are printed here
        answer = filter_answer(answer)
        print(f"SensorChat's answer: {answer}")
    print('Latency: ', end - decompose_start)

    new_log["question"] = question
    new_log["sensor_results"] = sensor_results
    new_log["model_answer"] = answer
    new_log["delay"] = end - decompose_start

    print_latency(time_profile)

    return jsonify({"bot_message": answer})


if __name__ == "__main__":
    HOST_IP = "132.239.17.132"
    HOST_PORT = 12345
    app.run(host=HOST_IP, debug=False, port=HOST_PORT)