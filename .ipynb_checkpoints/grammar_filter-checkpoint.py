import language_check as lck
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pickle as pkl
from IPython.core.debugger import set_trace


# open json and its part as specified e.g. key = 'annotations'
def open_json(vist_jsonpath, key):
    with open(vist_jsonpath) as f:
        json_dict = json.load(f)
        val = json_dict[key]
    return json_dict, val

def sentence_extract(val): #val from open json (dict)
    sen_list = []
    sen_proc_list = [] 
    for i in range(len(val)):
        sentence = val[i][0]['original_text']
        sentence_proc = val[i][0]['text']
        sen_list.append(sentence)
        sen_proc_list.append(sentence_proc)
    return sen_list, sen_proc_list

def ne_extract(sen_list):
    fin_ne_list = []
    for sen in sen_list:
        token_list = sen.split()
        for token in token_list:
            if token[0]=='[' and token[-1]==']':
                fin_ne_list.append(token)
    # remove duplicates
    fin_ne_list = list(set(fin_ne_list))
    return fin_ne_list

def sen_polish(sen_list):
    cap = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    small = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    char_dict= dict(zip(small, cap))
    
    fin_ne_list = ne_extract(sen_list)
    for idx, sen in enumerate(sen_list):
        sen_split = sen.split()
        #named entities replaced with Sam / i replaced with I 
        for i, token in enumerate(sen_split):
            if token in fin_ne_list:
                sen_split[i] = "Sam"
            if token == 'i':
                sen_split[i] = 'I'
                
        #whitespace in front of ".", "?", "!" are removed 
        ending_mark = sen[-1]
        sen = " ".join(sen_split)
        sen = sen[:-2]
        sen += ending_mark
        #capitalisation 
        first_char = sen[0]    
        if first_char in small:
            sen = sen[1:]
            sen = char_dict[first_char] + sen
        #isolated delimiters are replaced
        for delim in  ",", ".", "n't", "!", "?", ";", ":", "\'m", "\'ve", "\'d","\'s":
            target = " "+delim+" "
            if target in sen:
                piece_list = sen.split(target)
                sen = "{delim} ".format(delim=delim).join(piece_list)

        #re enter the polished sentence
        sen_list[idx] = sen
    return sen_list

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
            ok_arry.append(a_row)    
        if i%10000 == 0: 
            print("{idx}: {type}_df, adding a row below\n\t {row}".format(idx=i, type='ok' if len(instance)==0 else 'err',row=a_row))
    
    nparry_dict = {'ok': np.array(ok_arry), 'err': np.array(err_arry)}
    return nparry_dict


def make_df(nparry_dict): #input: make_np_arry output #output: pd.dataframes
    name_column = ['ds_idx', 'sentence', 'rule_id', 'msg', 'category', 'issuetype']
    ok_df = pd.DataFrame(nparry_dict['ok'], columns = name_column)
    err_df = pd.DataFrame(nparry_dict['err'], columns = name_column)
    df_dict = {'ok':ok_df, 'err':err_df}
    return df_dict 


