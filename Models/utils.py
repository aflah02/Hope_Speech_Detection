import json, os
from joblib import dump, load
from embeddings_loader import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

def get_f1_test_scores(filename):
    with open(filename, "r") as f:
        x = json.loads(f.read())
    result = []
    for each_cell in x["cells"]:
        if "outputs" in each_cell and each_cell["outputs"]:
            try:
                output = each_cell["outputs"][0]["text"]
                for each_output in output:
                    if "F1 Test" in each_output:
                        test_score = each_output.replace("F1 Test:  ", '')
                        test_score = test_score[:-2]
                        result.append(float(test_score))
            except:
                continue
    return result

def create_table(classifier, results):
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"{classifier}:", end = " ")
    for i in results[:-1]:
        print(f"{i} |", end = " ")
    print(f"{results[-1]}")

parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
save_folder_path = os.path.join(parent_dir, 'Models\Model Dumps')

def save_model(model, save_path):
    dump(model, os.path.join(save_folder_path, save_path))

def load_model(model_path):
    return load(os.path.join(save_folder_path, model_path))

train_labels, dev_labels, test_labels = load_labels()

label_replacement = {
    'Hope_speech': 0,
    'Non_hope_speech': 1,
    'not-English': 2,
}

# Replace labels with numbers
train_labels = [label_replacement[label] for label in train_labels]
dev_labels = [label_replacement[label] for label in dev_labels]
test_labels = [label_replacement[label] for label in test_labels]

def computeAllScores(y_pred_train, y_pred_dev, y_pred_test):
    print("Accuracy Train: ", accuracy_score(train_labels, y_pred_train))
    print("Accuracy Dev: ", accuracy_score(dev_labels, y_pred_dev))
    print("Accuracy Test: ", accuracy_score(test_labels, y_pred_test))
    print("Weighted F1 Train: ", f1_score(train_labels, y_pred_train, average='weighted'))
    print("Weighted F1 Dev: ", f1_score(dev_labels, y_pred_dev, average='weighted'))
    print("Weighted F1 Test: ", f1_score(test_labels, y_pred_test, average='weighted'))
    print("Macro F1 Train: ", f1_score(train_labels, y_pred_train, average='macro'))
    print("Macro F1 Dev: ", f1_score(dev_labels, y_pred_dev, average='macro'))
    print("Macro F1 Test: ", f1_score(test_labels, y_pred_test, average='macro'))
    print("Micro F1 Train: ", f1_score(train_labels, y_pred_train, average='micro'))
    print("Micro F1 Dev: ", f1_score(dev_labels, y_pred_dev, average='micro'))
    print("Micro F1 Test: ", f1_score(test_labels, y_pred_test, average='micro'))
    print("Weighted Recall Train: ", recall_score(train_labels, y_pred_train, average='weighted'))
    print("Weighted Recall Dev: ", recall_score(dev_labels, y_pred_dev, average='weighted'))
    print("Weighted Recall Test: ", recall_score(test_labels, y_pred_test, average='weighted'))
    print("Macro Recall Train: ", recall_score(train_labels, y_pred_train, average='macro'))
    print("Macro Recall Dev: ", recall_score(dev_labels, y_pred_dev, average='macro'))
    print("Macro Recall Test: ", recall_score(test_labels, y_pred_test, average='macro'))
    print("Micro Recall Train: ", recall_score(train_labels, y_pred_train, average='micro'))
    print("Micro Recall Dev: ", recall_score(dev_labels, y_pred_dev, average='micro'))
    print("Micro Recall Test: ", recall_score(test_labels, y_pred_test, average='micro'))
    # Confusion Matrix
    print("Confusion Matrix Train: ")
    print(confusion_matrix(train_labels, y_pred_train))
    print("Confusion Matrix Dev: ")
    print(confusion_matrix(dev_labels, y_pred_dev))
    print("Confusion Matrix Test: ")
    print(confusion_matrix(test_labels, y_pred_test))

if __name__=="__main__":
    for filename in os.listdir():
        if ".ipynb" in filename:
            create_table(filename.replace(".ipynb", ''), get_f1_test_scores(filename))