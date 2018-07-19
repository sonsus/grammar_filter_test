import language_check as lck
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pickle as pkl



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

'''
def error_checker(sen_list):
    tool = lck.LanguageTool('en-US')
    err_type_ex_dict = {}
    for i,sen in enumerate(sen_list):
        instance = tool.check(sen)
        if len(instance)>0: #if there is no such an grammatical err, len == 0
            err_type_ex_dict[instance.ruleId]['num']
            if instance.ruleId not in list(err_type_ex_dict.keys()):
                err_type_ex_dict[instance.ruleId] = {'num'=i , 'instances'= []}
            err_type_ex_dict[instance.ruleId]
'''

def make_np_arry(sen_list): #return dict  {'ok': oknparry, 'err': errnparry}
    #make list first and at the last, convert them by np.array() 
    ok_arry = []
    err_arry = []

    tool = lck.LanguageTool('en-US')
    for i, sen in enumerate(sen_list):
        instance = tool.check(sen)
        if len(instance) > 0:
            for j in range(len(instance)):
                rule_id = instance[j].ruleId
                msg = instance[j].msg
                category = instance[j].category
                issuetype = instance[j].locqualityissuetype
                
                a_row = [i, sen, rule_id, msg, category, issuetype]
                err_arry.append(a_row)
                
        else: # that is len(instance)==0, no error!
            rule_id = 'N/A'
            msg = 'N/A'
            category = 'N/A'
            issuetype = 'N/A'        
            a_row = [i, sen, rule_id, msg, category, issuetype]
            ok_arry.append()    
    
    nparry_dict = {'ok': np.array(ok_arry), 'err': np.array(err_arry)}
    return nparry_dict


def make_df(nparry_dict): #input: make_np_arry output #output: pd.dataframes
    name_column = ['ds_idx', 'sentence', 'rule_id', 'msg', 'category', 'issuetype']
    ok_df = pd.DataFrame(nparry_dict['ok'], columns = name_column)
    err_df = pd.DataFrame(nparry_dict['err'], columns = name_column)
    df_dict = {'ok':ok_df, 'err':err_df}
    return df_dict 

            

#json paths 
split = ['train', 'val', 'test']
path = ["./sis/{split}.story-in-sequence.json".format(split = spl) for spl in split]
sis_path_dict =  dict(zip(split, path))

# grammar checker
tool = lck.LanguageTool('en-US')

# json dict and annotations
#train_dict, train_annot = open_json(sis_path_dict['train'], 'annotations')
#val_dict, val_annot = open_json(sis_path_dict['val'], 'annotations')
test_dict, test_annot = open_json(sis_path_dict['test'], 'annotations')

# sentence extraction 
#train_sen_list = sentence_extract(train_annot)
#val_sen_list = sentence_extract(val_annot)
test_sen_list = sentence_extract(test_annot)

# grammar check
#print("train")
#train_err_list = error_checker(train_sen_list)
#print("val")
#val_err_list = error_checker(val_sen_list)
print("test")


# save the dict binary with pkl
# when loading pkl, we need "language_check" package imported
nparry_dict = make_np_arry(test_sen_list)
df_dict = make_df(nparry_dict)
with open("test_sis_grammar.pkl", "wb") as f:
    pkl.dump(df_dict, f)

