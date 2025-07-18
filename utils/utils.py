import numpy as np
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz
import re
import random

full_time_of_day = ['morning', 'afternoon', 'evening', 'night', 'noon',
                    'early morning', 'late morning', 'early afternoon',
                    'late afternoon', 'early evening', 'late evening']

def similarity_ratio(str1, str2):
    return fuzz.partial_ratio(str1, str2)


def get_start_end_unix_time(date_str, time_of_day=''):
    
    # Parse the date string into a datetime object
    dt = datetime.strptime(date_str, "%Y-%m-%d %A")
    
    # Calculate the start of the day (midnight)
    start = start_of_day = datetime(dt.year, dt.month, dt.day)
    
    # Calculate the end of the day (just before midnight of the next day)
    end = start_of_day + timedelta(days=1) - timedelta(seconds=1)

    if len(time_of_day) > 0:
        # time_of_day must be one of 'morning', 'afternoon', 'evening', 'night', 'noon' or empty
        time_of_day = time_of_day.lower()
        assert time_of_day in full_time_of_day, \
            f"Unexpected time of day {time_of_day}!"
        
        if time_of_day == 'morning':
            start = start_of_day + timedelta(hours=6)  # 6am
            end = start_of_day + timedelta(hours=12)  # 12pm
        elif time_of_day == 'noon':
            start = start_of_day + timedelta(hours=11)  # 11am
            end = start_of_day + timedelta(hours=13)  # 1pm
        elif time_of_day == 'afternoon':
            start = start_of_day + timedelta(hours=12)  # 12pm
            end = start_of_day + timedelta(hours=17)  # 5pm
        elif time_of_day == 'evening':
            start = start_of_day + timedelta(hours=17)  # 5pm
            end = start_of_day + timedelta(hours=21)  # 9pm
        elif time_of_day == 'night':
            start = start_of_day + timedelta(hours=21)  # 9pm
            end = start_of_day + timedelta(hours=30)  # 6am next day

        elif time_of_day == 'early morning':
            start = start_of_day + timedelta(hours=6)  # 6am
            end = start_of_day + timedelta(hours=8)  # 8am
        elif time_of_day == 'late morning':
            start = start_of_day + timedelta(hours=10)  # 10am
            end = start_of_day + timedelta(hours=12)  # 12pm
        elif time_of_day == 'early afternoon':
            start = start_of_day + timedelta(hours=12)  # 12am
            end = start_of_day + timedelta(hours=14)  # 2pm
        elif time_of_day == 'late afternoon':
            start = start_of_day + timedelta(hours=15)  # 3pm
            end = start_of_day + timedelta(hours=17)  # 5pm
        elif time_of_day == 'early evening':
            start = start_of_day + timedelta(hours=17)  # 5pm
            end = start_of_day + timedelta(hours=19)  # 7pm
        elif time_of_day == 'late evening':
            start = start_of_day + timedelta(hours=19)  # 7pm
            end = start_of_day + timedelta(hours=21)  # 9pm
        else:
            raise ValueError

    #print('start: ', start.strftime('%Y-%m-%d %H:%M:%S'))
    #print('end: ', end.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Convert to Unix timestamps
    start_unix_time = int(start.timestamp())
    end_unix_time = int(end.timestamp())
    
    return start_unix_time, end_unix_time


def unix_to_time_string(unix_time):
    # Convert Unix timestamp to a datetime object
    dt = datetime.fromtimestamp(unix_time)
    # Format the datetime object to the desired string format
    time_string = dt.strftime("%H:%M%p").lower()
    return time_string


def smooth_pred(T, Y, window_size=5):
    # window size defines the smooth time range
    # The time range considered is +-window_size in minutes
    Y_new = np.zeros_like(Y)
    for cls in range(Y.shape[1]):
        for idx, t in enumerate(T):
            delta = window_size*60
            mask_low = T > t - delta
            mask_high = T < t + delta
            mask = mask_low & mask_high

            Y_new[idx, cls] = np.mean(Y[:, cls][mask])

    return Y_new


def match_list(activity, all_activities):
    # match activity with the most similar one in all_activities
    # If exact match, then directly use that item
    if activity in all_activities:
        return activity, all_activities.index(activity)
    
    # If not, use the similarity score
    #print('original activity: ', activity)
    simil_scores = list(map(lambda s: similarity_ratio(s.lower(), activity.lower()), all_activities))
    matched_act_idx = np.argmax(np.array(simil_scores))
    matched_activity = all_activities[matched_act_idx]
    #print('matched activity: ', matched_activity)
    return matched_activity, matched_act_idx


def match_date(date, all_dates):
    # match date with the most similar one in all_dates
    #print('original date: ', date)
    simil_scores = list(map(lambda s: similarity_ratio(s.lower(), date.lower()), all_dates))
    matched_date_idx = np.argmax(np.array(simil_scores))
    matched_date = all_dates[matched_date_idx]
    #print('matched date: ', matched_date)
    return matched_date, matched_date_idx


def count_edges(sequence, conseq=5):
    # It needs to remain 1 for at least conseq consecutive times
    count = 0
    i = 0
    n = len(sequence)
    
    while i < n - conseq:
        if sequence[i] == 0 and all(sequence[i + j] == 1 for j in range(1, conseq+1)):
            count += 1
            i += conseq+1  # Skip ahead to avoid double counting
        else:
            i += 1
    
    return count


def extract_function_name(string):
    # Define a regular expression pattern to match content within '<<>>'
    pattern = re.compile(r'<<(.*?)>>')
    
    # Find all matches in the string
    matches = pattern.findall(string)
    
    return list(set(matches))


def extract_activity(string):
    # Define a regular expression pattern to match content within '<<>>'
    pattern = re.compile(r'\(\((.*?)\)\)')
    
    # Find all matches in the string
    matches = pattern.findall(string)
    
    return list(set(matches))


def extract_date(string):
    # Define a regular expression pattern to match content within '<<>>'
    pattern = re.compile(r'\[\[(.*?)\]\]')
    
    # Find all matches in the string
    matches = pattern.findall(string)
    
    return list(set(matches))


def extract_time_of_day(string):
    # Define a regular expression pattern to match content within '<<>>'
    pattern = re.compile(r'\|\|(.*?)\|\|')
    
    # Find all matches in the string
    matches = pattern.findall(string)
    
    return list(set(matches))


def find_most_frequent_element(strings):
    if not strings:
        return None

    count = {}

    for string in strings:
        if string in count:
            count[string] += 1
        else:
            count[string] = 1

    max_count = max(count.values())
    candidates = [string for string, freq in count.items() if freq == max_count]

    return random.choice(candidates)
