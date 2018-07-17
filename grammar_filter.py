import language_check as lck
import json
from pathlib import Path



# open json and its part as specified e.g. key = 'annotations'
def open_json(vist_jsonpath, key):
    with open(vist_jsonpath) as f:
        json_dict = json.load(f)
        val = json_dict[key]
    return json_dict, val

def sentence_extract(val): #val from open json (dict)
    sen_list = []
    for i in range(len(val)):
        sentence = val[i][0]['original_text']
        sen_list.append(sentence)
    return sen_list

def error_checker(sen_list):
    tool = lck.LanguageTool('en-US')
    err_log_list = []
    for i,sen in enumerate(sen_list):
        if len(tool.check(sen))>0:
            err_log_list.append( (i,sen) )
            print("{}: err found".format(i))
    return err_log_list

#json paths 
split = ['train', 'val', 'test']
path = ["./sis/{split}.story-in-sequence.json".format(split = spl) for spl in split]
sis_path_dict =  dict(zip(split, path))

# grammar checker
tool = lck.LanguageTool('en-US')

# json dict and annotations
train_dict, train_annot = open_json(sis_path_dict['train'], 'annotations')
val_dict, val_annot = open_json(sis_path_dict['val'], 'annotations')
test_dict, test_annot = open_json(sis_path_dict['test'], 'annotations')

# sentence extraction 
train_sen_list = sentence_extract(train_annot)
val_sen_list = sentence_extract(val_annot)
test_sen_list = sentence_extract(test_annot)

# grammar check
print("train")
train_err_list = error_checker(train_sen_list)
print("val")
val_err_list = error_checker(val_sen_list)
print("test")
test_err_list = error_checker(test_sen_list)

print(train_err_list)
print(val_err_list)
print(test_err_list)
