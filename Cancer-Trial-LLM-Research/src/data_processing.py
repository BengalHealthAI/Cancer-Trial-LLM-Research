import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_process_data(input_path):
    # Read raw data (assume tab-delimited CSV)
    dataset = pd.read_table(input_path, header=None, delimiter='\t')
    label_list, study_list, condition_list = [], [], []
    for i, row in dataset.iterrows():
        row_split = row[0].split('\t')
        label_list.append(row_split[0])
        study_condition_split = row_split[1].split('.')
        study_list.append(study_condition_split[0])
        condition_list.append(study_condition_split[1])
    df = pd.DataFrame({'Label': label_list, 'Study': study_list, 'Condition': condition_list})
    df['Condition'] = df['Condition'].str.rstrip(',')
    return df

def split_data(df, test_size=0.3, random_state=42):
    train_data, test_data = train_test_split(
        df, test_size=test_size, stratify=df['Label'], random_state=random_state
    )
    return train_data, test_data

def transform_to_llama2_format(example, sample_type='train'):
    study_text = example.get('Study', '').strip()
    condition_text = example.get('Condition', '').strip()
    label_text = example.get('Label', '').strip()
    if not study_text or not condition_text or not label_text:
        return None
    system_message = (
        "Interventional cancer clinical trials are generally too restrictive, and some patients are often excluded "
        "based on comorbidity, past or concomitant treatments, or age. In this work, we built a model to predict "
        "whether short clinical statements were considered inclusion or exclusion criteria.\n\n"
        "You are an assistant who will help decide whether a statement is inclusion or exclusion criteria. "
        "Examples: '__label__0' for Eligible and '__label__1' for Not Eligible.\n\n"
        "Eligibility criteria are split into 'Study Intervention' and 'Condition'."
    )
    human_prompt = f"Study Intervention: {study_text}\nCondition: {condition_text}"
    if sample_type == 'train':
        return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{human_prompt} [/INST] {label_text} </s>"
    else:
        return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{human_prompt} [/INST]"

if __name__ == '__main__':
    # Example usage:
    df = load_and_process_data("data/labeledEligibilitySample1000000.csv")
    train_df, test_df = split_data(df)
    train_df['llama2_message'] = train_df.apply(lambda row: transform_to_llama2_format(row, 'train'), axis=1)
    test_df['llama2_message'] = test_df.apply(lambda row: transform_to_llama2_format(row, 'test'), axis=1)
    train_df.to_pickle("data/train_data_llama2.pkl")
    test_df.to_pickle("data/test_data_llama2.pkl")
