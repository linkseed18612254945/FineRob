import random
import pandas as pd

def shuffle_options(options):
    options = eval(options)
    right_answer = options[0]
    random.shuffle(options)
    right_index = options.index(right_answer)
    return options, right_index

df = pd.read_csv('data/mini_reddit_qa.csv')
content_df = df[df['qtype'] == 'action_content']
other_df = df[df['qtype'] != 'action_content']

all_new_options = []
all_right_index = []
for options in content_df['choices'].tolist():
    new_options, right_index = shuffle_options(options)
    all_new_options.append(new_options)
    all_right_index.append(right_index)
content_df['choices'] = all_new_options
content_df['answer_index'] = all_right_index

new_df = pd.concat([content_df, other_df], ignore_index=True)
new_df = new_df.sample(frac=1)
new_df.to_csv('data/mini_reddit_qa.csv', index=False)
