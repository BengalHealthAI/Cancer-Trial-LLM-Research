import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def calculate_metrics(df, true_label_col='Label', predicted_label_col='prediction'):
    true_labels = df[true_label_col]
    predicted_labels = df[predicted_label_col]
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

if __name__ == '__main__':
    df = pd.read_pickle("data/test_data_llama2_predicted.pkl")
    metrics = calculate_metrics(df)
    print(metrics)
