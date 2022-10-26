import json, os

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

if __name__=="__main__":
    for filename in os.listdir():
        if ".ipynb" in filename:
            create_table(filename.replace(".ipynb", ''), get_f1_test_scores(filename))