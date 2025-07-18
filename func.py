import os
import json
import argparse
import numpy as np
from typing import List

from utils.es_data_utils import get_label_pretty_name, merged_label_names
from utils.es_data_utils import all_user_ids

from utils.utils import match_list, match_date
from utils.utils import smooth_pred, count_edges
from utils.utils import get_start_end_unix_time, unix_to_time_string

def calculate_duration(data_Y: np.ndarray, 
                       data_T: np.ndarray,
                       activity: str, 
                       all_activities: List[str], 
                       date: str, 
                       all_dates: List[str],
                       time_of_day: str,
                       thres: int=10):
    #### activity should be a str, either one concrete activity or 'all activities'
    #### date should be a str, either a concrete day or 'all days'
    # make the data smooth
    data_Y = smooth_pred(data_T, data_Y, window_size=3)

    # get activity index
    if activity == 'all activities':
        matched_activity = all_activities
        matched_act_idx = np.arange(len(all_activities))
    else:
        matched_activity, matched_act_idx = match_list(activity, all_activities)
        matched_activity, matched_act_idx = [matched_activity], [matched_act_idx]

    if date in ['all days', 'last week']: # Need to iterate through all days
        search_dates = all_dates
    else:  # Only search the given date
        search_dates = [date]
    
    result = ""
    for d in search_dates:
        # get date time range
        matched_date, match_date_idx = match_date(d, all_dates)
        
        # get start and end time of the date
        start_unix_time, end_unix_time = get_start_end_unix_time(matched_date, time_of_day)

        # get the T that falls in this range
        mask_1 = data_T > start_unix_time
        mask_2 = data_T < end_unix_time
        mask = mask_1 & mask_2

        # get the final time results
        for act, act_idx in zip(matched_activity, matched_act_idx):
            minutes = data_Y[mask, act_idx].sum()
            if minutes // 60 > 0:
                if len(time_of_day) > 0:
                    result += f"You spent {minutes // 60} hours and {minutes % 60} minutes {act} in the {time_of_day} on {matched_date}. "
                else:
                    result += f"You spent {minutes // 60} hours and {minutes % 60} minutes {act} on {matched_date}. "
            else:
                if len(time_of_day) > 0:
                    result += f"You spent {minutes % 60} minutes {act} in the {time_of_day} on {matched_date}. "
                else:
                    result += f"You spent {minutes % 60} minutes {act} on {matched_date}. "

    return result


def calculate_days(data_Y: np.ndarray, 
                   data_T: np.ndarray,
                   activity: str, 
                   all_activities: List[str], 
                   date: str,  # NOT USED
                   all_dates: List[str],
                   time_of_day: str,  # NOT USED
                   thres: int=10):
    # thres is a threshold for judging whether this activity happens
    # make the data smooth
    data_Y = smooth_pred(data_T, data_Y, window_size=3)

    # get activity index
    matched_activity, matched_act_idx = match_list(activity, all_activities)
    
    cnt = 0
    for d in all_dates:
        # get date time range
        matched_date, match_date_idx = match_date(d, all_dates)
        
        # get start and end time of the date
        start_unix_time, end_unix_time = get_start_end_unix_time(matched_date)

        # get the T that falls in this range
        mask_1 = data_T > start_unix_time
        mask_2 = data_T < end_unix_time
        mask = mask_1 & mask_2

        # get the final time results
        minutes = data_Y[mask, matched_act_idx].sum()
        #print(minutes)
        if minutes > thres:
            cnt += 1 # Add one more day
    
    digit_words = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine'
    }
    
    if cnt < 2:
        result = f"You were {matched_activity} {digit_words[cnt]} day. "
    elif cnt < 10:
        result = f"You were {matched_activity} {digit_words[cnt]} days. "
    else:
        result = f"You were {matched_activity} {cnt} days. "

    return result


def calculate_frequency(data_Y: np.ndarray, 
                        data_T: np.ndarray,
                        activity: str, 
                        all_activities: List[str], 
                        date: str, 
                        all_dates: List[str],
                        time_of_day: str):
    #### date can be a string, or None (search on all dates)
    # make the data smooth
    data_Y = smooth_pred(data_T, data_Y, window_size=5)

    # get activity index
    # get activity index
    if activity == 'all activities':
        matched_activity = all_activities
        matched_act_idx = np.arange(len(all_activities))
    else:
        matched_activity, matched_act_idx = match_list(activity, all_activities)
        matched_activity, matched_act_idx = [matched_activity], [matched_act_idx]

    if date in ['all days', 'last week']: # Need to iterate through all days
        search_dates = all_dates
    else:  # Only search the given date
        search_dates = [date]
    
    result = ""
    for d in search_dates:
        # get date time range
        matched_date, match_date_idx = match_date(d, all_dates)
        
        # get start and end time of the date
        start_unix_time, end_unix_time = get_start_end_unix_time(matched_date, time_of_day)

        # get the T that falls in this range
        mask_1 = data_T > start_unix_time
        mask_2 = data_T < end_unix_time
        mask = mask_1 & mask_2

        # get the final frequency results
        for act, act_idx in zip(matched_activity, matched_act_idx):
            edges = count_edges(data_Y[mask, act_idx])
            if edges > 1:
                if len(time_of_day) > 0:
                    result += f"You were {act} for {edges} times in the {time_of_day} on {matched_date}. "
                else:
                    result += f"You were {act} for {edges} times on {matched_date}. "
            else:
                if len(time_of_day) > 0:
                    result += f"You were {act} for {edges} time in the {time_of_day} on {matched_date}. "
                else:
                    result += f"You were {act} for {edges} time on {matched_date}. "

    return result


def detect_first_time(data_Y: np.ndarray, 
                      data_T: np.ndarray,
                      activity: str, 
                      all_activities: List[str], 
                      date: str, 
                      all_dates: List[str],
                      time_of_day: str):
    #### date must be a string!
    # make the data smooth
    data_Y = smooth_pred(data_T, data_Y, window_size=2) # Usually first time is more accurate

    # get activity index
    if activity == 'all activities':
        matched_activity = all_activities
        matched_act_idx = np.arange(len(all_activities))
    else:
        matched_activity, matched_act_idx = match_list(activity, all_activities)
        matched_activity, matched_act_idx = [matched_activity], [matched_act_idx]

    # get date time range
    matched_date, match_date_idx = match_date(date, all_dates)
    
    # get start and end time of the date
    start_unix_time, end_unix_time = get_start_end_unix_time(matched_date, time_of_day)

    # get the T that falls in this range
    mask_1 = data_T > start_unix_time
    mask_2 = data_T < end_unix_time
    mask = mask_1 & mask_2

    # get the final time results
    result = ""
    for act, act_idx in zip(matched_activity, matched_act_idx):
        if data_Y[mask, act_idx].sum() > 0:
            first_idx = np.argmax(data_Y[mask, act_idx])
            formatted_time = unix_to_time_string(data_T[mask][first_idx])

            if len(time_of_day) > 0:
                result += f"You were {act} first time at {formatted_time} in the {time_of_day} on {matched_date}. "
            else:
                result += f"You were {act} first time at {formatted_time} on {matched_date}. "
        
        else:
            if len(time_of_day) > 0:
                result += f"You were not {act} in the {time_of_day} on {matched_date}. "
            else:
                result += f"You were not {act} on {matched_date}. "

    return result


def detect_last_time(data_Y: np.ndarray, 
                     data_T: np.ndarray,
                     activity: str, 
                     all_activities: List[str], 
                     date: str, 
                     all_dates: List[str],
                     time_of_day: str):
    #### date must be a string!
    # make the data smooth
    data_Y = smooth_pred(data_T, data_Y, window_size=3)

    # get activity index
    matched_activity, matched_act_idx = match_list(activity, all_activities)

    # get date time range
    matched_date, match_date_idx = match_date(date, all_dates)
    
    # get start and end time of the date
    start_unix_time, end_unix_time = get_start_end_unix_time(matched_date, time_of_day)

    # get the T that falls in this range
    mask_1 = data_T > start_unix_time
    mask_2 = data_T < end_unix_time
    mask = mask_1 & mask_2

    # get the final time results
    if data_Y[mask, matched_act_idx].sum() > 0:
        last_idx = np.argmax(data_Y[mask, matched_act_idx][::-1])
        formatted_time = unix_to_time_string(data_T[mask][::-1][last_idx])

        if len(time_of_day) > 0:
            result = f"You were {matched_activity} last time at {formatted_time} in the {time_of_day} on {matched_date}. "
        else:
            result = f"You were {matched_activity} last time at {formatted_time} on {matched_date}. "
    
    else:
        if len(time_of_day) > 0:
            result = f"You were not {matched_activity} in the {time_of_day} on {matched_date}. "
        else:
            result = f"You were not {matched_activity} on {matched_date}. "

    return result


def find_activity(data_Y: np.ndarray, 
                  data_T: np.ndarray,
                  activity: str,
                  all_activities: List[str], 
                  date: str,
                  all_dates: List[str],
                  time_of_day: str,
                  thres: int=10): 
    # thres is a threshold for judging whether this activity happens
    #### date must be a string!
    # make the data smooth
    data_Y = smooth_pred(data_T, data_Y, window_size=2)

    # get date time range
    matched_date, match_date_idx = match_date(date, all_dates)
    
    # get start and end time of the date
    start_unix_time, end_unix_time = get_start_end_unix_time(matched_date, time_of_day)

    # get the T that falls in this range
    mask_1 = data_T > start_unix_time
    mask_2 = data_T < end_unix_time
    mask = mask_1 & mask_2

    # get the activities that happened
    act_vec = data_Y[mask].sum(axis=0)
    
    #print(act_vec)
    act_str = ""
    for i in range(len(all_activities)):
        if act_vec[i] > thres:
            if len(act_str) > 0:
                act_str += ", "

            act_str += all_activities[i]

    if len(time_of_day) > 0:
        result = f"You were {act_str} in the {time_of_day} on {matched_date}. "
    else:
        result = f"You were {act_str} on {matched_date}. "

    return result


"""def find_cooccurence_activity(data_Y: np.ndarray, 
                              data_T: np.ndarray,
                              co_activity: str,
                              all_activities: List[str], 
                              date: str,
                              all_dates: List[str],
                              thres: int=10): 
    # thres is a threshold for judging whether this activity happens
    # co_activity: the question is asking about the activity 
    # happening at the same time of this co_activity
    #### date must be a string!
    # make the data smooth
    data_Y = smooth_pred(data_T, data_Y, window_size=2)

    # get date time range
    matched_date, match_date_idx = match_date(date, all_dates)
    
    # get start and end time of the date
    start_unix_time, end_unix_time = get_start_end_unix_time(matched_date)

    # get the T that falls in this range
    mask_1 = data_T > start_unix_time
    mask_2 = data_T < end_unix_time
    mask = mask_1 & mask_2


    # if co_activity is given, only focusing on the activities that happened 
    matched_activity, matched_act_idx = match_activity(co_activity, all_activities)
    co_act_mask = data_Y[mask, matched_act_idx].astype(bool)
    act_vec = data_Y[mask][co_act_mask].sum(axis=0)
    
    #print(act_vec)
    act_str = ""
    for i in range(len(all_activities)):
        if act_vec[i] > thres and all_activities[i] != co_activity:
            if len(act_str) > 0:
                act_str += ", "

            act_str += all_activities[i]

    result = f"You were {act_str} while you were {matched_activity} on {matched_date}. "

    return result"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="the path to ExtraSensory.per_uuid_features_labels")
    args = parser.parse_args()

    user_idx = 31
    user_id = all_user_ids[user_idx]
    data = np.load(os.path.join(args.data_path, '{}.npz'.format(user_id)))

    all_dates = json.load(open('full_dates.json'))
    all_activities = list(map(get_label_pretty_name, merged_label_names))
    print(all_activities)
    print(all_dates[str(user_idx)])

    candidate_labels = ['at home', 'talking', 'with friends', 'with co-workers']
    ind = np.array([all_activities.index(a) for a in candidate_labels])

    result = calculate_duration(data['Y'][:, ind], data['T'], 'at home', candidate_labels, 'all days', all_dates[str(user_idx)], '')
    print(result)
    """result = calculate_frequency(data['Y'], data['T'], 'eat', all_activities, 'all days', all_dates[str(user_idx)])
    print(result)
    result = detect_first_time(data['Y'], data['T'], 'groom', all_activities, 'Friday', all_dates[str(user_idx)])
    print(result)
    result = detect_last_time(data['Y'], data['T'], 'run', all_activities, 'Wednesday', all_dates[str(user_idx)])
    print(result)
    result = find_activity(data['Y'], data['T'],  all_activities, 'Wednesday', all_dates[str(user_idx)])
    print(result)
    result = calculate_days(data['Y'], data['T'], 'meeting', all_activities, all_dates[str(user_idx)])
    print(result)"""