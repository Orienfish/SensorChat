import re

def contains_weekday(sentence):
    # Define a list of weekday names
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Create a regular expression pattern to match any of the weekdays
    pattern = re.compile(r'\b(?:' + '|'.join(weekdays) + r')\b', re.IGNORECASE)
    
    # Search for the pattern in the sentence
    match = pattern.search(sentence)
    
    # Return True if a match is found, else False
    return bool(match)


def clean_date_info(question, answer, today_date):
    # replace yesterday or today with the actual date info
    if 'yesterday' in question:
        question = question.replace('yesterday', f'on {today_date}')
    elif 'Yesterday' in question:
        question = question.replace('Yesterday', f'On {today_date}')
    

    if 'today' in question:
        question = question.replace('today', f'on {today_date}')
    elif 'Today' in question:
        question = question.replace('Today', f'On {today_date}')

    if 'this week' in question:
        question = question.replace(' this week', '')
    elif 'This week' in question:
        question = question.replace('This week', '')
    
    if 'yesterday' in answer:
        answer = answer.replace('yesterday', f'on {today_date}')
    elif 'Yesterday' in answer:
        answer = answer.replace('Yesterday', f'On {today_date}')

    if 'today' in answer:
        answer = answer.replace('today', f'on {today_date}')
    elif 'Today' in answer:
        answer = answer.replace('Today', f'On {today_date}')

    if 'this week' in answer:
        answer = answer.replace(' this week', '')
    elif 'This week' in answer:
        answer = answer.replace('This week', '')

    # for daily questions, detect if there are week day information in the question
    # if none, then add today's date
    if len(today_date) > 0 and not contains_weekday(question):
        # add today's date to question
        question = question.split('?')[0] + f' on {today_date}?'
    
    return question, answer

