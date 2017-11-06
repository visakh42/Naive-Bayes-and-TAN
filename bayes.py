# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:57:14 2017

@author: visakh
"""

import sys
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

def readdata(file_to_read):
    dataset = arff.loadarff(file_to_read)
    data = pd.DataFrame(dataset[0])
    for i in range(0,len(data['class'])):
        for j in range (0,data.shape[1]):
            data.loc[i][j] = data.loc[i][j].decode()
    return data

def freq(train_data,attr_names,attr_vals):
    global class_identify
    freq_data = {}
    freq_data[0] = {}
    freq_data[1] = {}
    for i in attr_vals['class']:
        freq_data[i] = {}
        train_set = train_data[train_data['class']==i]
        for j in attr_names[:-1]:     
            freq_data[i][j] = train_set[j].value_counts()
    return freq_data

def calculate_probs(train_data,attr_names,attr_vals):
    data_freq = freq(train_data,attr_names,attr_vals)
    prob_of_class = {}
    prob_of_feature_value_given_class = {}
    for i in attr_vals['class']:
        prob_of_class[i] = train_data[train_data['class']==i].shape[0]/train_data.shape[0]
    for i in attr_vals['class']:
        prob_of_feature_value_given_class[i] = {}
        for j in attr_names[:-1]:
            prob_of_feature_value_given_class[i][j] = {}
            total_in_class = 0
            uniques_in_set = train_data[j].unique()
            total_uniques_in_set = len(uniques_in_set)
            for k in attr_vals[j]:
                if k in data_freq[i][j]:
                    total_in_class += data_freq[i][j][k]
                else:
                    data_freq[i][j][k] = 0                
            for l in attr_vals[j]: 
                prob_of_feature_value_given_class[i][j][l] = (data_freq[i][j][l]+1)/(train_data[train_data['class']==i].shape[0]+1+total_uniques_in_set)
    return prob_of_feature_value_given_class,prob_of_class
                
        
def predict(test_data,prob_of_feature_value_given_class,prob_of_class,attr_names,attr_vals,train_data):
    predictions = "***** Naive Bayes Predcitions ******* \n S.No  prediction  Actual  Probability"
    predict = {}
    global naive_accuracy
    count = 0
    total = 0
    predict_feature_set_given_class = {}
    for i in range(0,test_data.shape[0]):
        for l in attr_vals['class']:
            predict[l] = 1
            predict_feature_set_given_class[l] = 1
        for j in attr_names[:-1]:
            for k in attr_vals['class']:
                if test_data[j].iloc[i] in prob_of_feature_value_given_class[k][j]:
                    predict_feature_set_given_class[k] *= prob_of_feature_value_given_class[k][j][test_data[j].iloc[i]]
                else:
                    predict_feature_set_given_class[k] *= 1/(train_data[train_data['class']==i].shape[0]+1+len(train_data[j].unique()))
        evidence = 0
        for n in attr_vals['class']: 
            evidence += predict_feature_set_given_class[n]*prob_of_class[n]            
        for m in attr_vals['class']:
            predict[m] = predict_feature_set_given_class[m]*prob_of_class[m]/evidence
        predictions += str (i+1) + " " + str(max(predict, key=predict.get)) + " " + str(test_data['class'].iloc[i]) + " " + str('{0:.12f}'.format(max(predict.values()))) + "\n"
        total+=1
        if(max(predict, key=predict.get) == str(test_data['class'].iloc[i])):
            count += 1
    print(count)
    naive_accuracy.append(count/total)
    
    return predictions    
    

def summarize(train_data):
    attr_names = list(train_data)
    attr_vals = {}
    for i in attr_names:
        attr_vals[i] = train_data[i].unique()
    return attr_names,attr_vals

def maximal_weight(attr_names,attr_vals,mset,train_data):
    best_for_i = 0
    for i in attr_names[:-1]:
        if i not in mset:
            best_for_j = 0
            for j in mset:
                likelihood_for_prim = 0
                for k in attr_vals[i]:
                    for l in attr_vals[j]:
                        for m in attr_vals['class']:
                            prob_i_j_y = (train_data[(train_data[i]==k) & (train_data[j]==l) & (train_data['class']==m)].shape[0]+1)/(train_data.shape[0]+1+ len(train_data[i].unique()) + len(train_data[j].unique()))
                            prob_i_j_given_y = (train_data[(train_data[i]==k) & (train_data[j]==l) & (train_data['class']==m)].shape[0]+1)/(train_data[train_data['class']==m].shape[0]+1+ len(train_data[i].unique()) + len(train_data[j].unique()))
                            prob_i_given_y = (train_data[(train_data[i]==k) & (train_data['class']==m)].shape[0] + 1)/(train_data[train_data['class']==m].shape[0] + 1 + len(train_data[i].unique()))
                            prob_j_given_y = (train_data[(train_data[j]==l) & (train_data['class']==m)].shape[0] + 1)/(train_data[train_data['class']==m].shape[0] + 1 + len(train_data[j].unique()))
                            logs = np.log2(prob_i_j_given_y/(prob_i_given_y*prob_j_given_y))
                            likelihood_for_prim += ( prob_i_j_y * logs)
                if(likelihood_for_prim > best_for_j):
                    best_for_j = likelihood_for_prim 
                    best_feature_in_j = j
            if(best_for_j > best_for_i):
                best_for_i = best_for_j
                child = i
                parent = best_feature_in_j
    return(parent,child)
                

def prim(attr_names,attr_vals,train_data):
    mset = []
    parents = {}
    mset.append(attr_names[0])
    bayesian_tree = {}
    parents[attr_names[0]] = attr_names[-1]
    while len(mset) < (len(attr_names) - 1):
        parent,child = maximal_weight(attr_names,attr_vals,mset,train_data)
        mset.append(child)
        bayesian_tree[child]=parent
    return bayesian_tree


def tan_predict(attr_names,attr_vals,train_data,test_data,bayesian_network):
    tanprediction = "***** TAN Predcitions ******* \n S.No  prediction  Actual  Probability \n"
    global tan_accuracy
    total = 0
    likelihood = {}
    tanpredict = {}
    prob_of_class = {}
    tan_count = 0
    data_freq = freq(train_data,attr_names,attr_vals)
    for o in attr_vals['class']:
        prob_of_class[o] = train_data[train_data['class']==o].shape[0]/train_data.shape[0]
    for i in range(0,test_data.shape[0]):
        for l in attr_vals['class']:
            likelihood[l] = 1
            tanpredict[l] = 1
        for j in attr_names[:-1]:
            for m in attr_vals['class']: 
                if j == attr_names[0]:
                    if test_data[j].iloc[i] in data_freq[m][j]: 
                        num = data_freq[m][j][test_data[j].iloc[i]]+1
                        den = train_data[train_data['class']==m].shape[0] + 1 +len(train_data[j].unique())
                        prob_feature_given_parent_and_class = num/den
                    else:
                        prob_feature_given_parent_and_class = 1/(train_data[train_data['class']==m].shape[0] + 1 +len(train_data[j].unique()))
                else:
                    if (test_data[j].iloc[i] in data_freq[m][j]) and (test_data[bayesian_network[j]].iloc[i] in data_freq[m][bayesian_network[j]]):
                        num = train_data[(train_data[bayesian_network[j]]== test_data[bayesian_network[j]].iloc[i]) & (train_data[j]==test_data[j].iloc[i]) & (train_data['class']==m)].shape[0] + 1
                        den = train_data[(train_data[bayesian_network[j]]== test_data[bayesian_network[j]].iloc[i]) & (train_data['class']==m)].shape[0] + 1 +len(train_data[j].unique())
                    else:
                        if test_data[j].iloc[i] in data_freq[m][j]:
                            num = data_freq[m][j][test_data[j].iloc[i]]+1
                            den = train_data[train_data['class']==m].shape[0] + 1 +len(train_data[j].unique())
                            prob_feature_given_parent_and_class = num/den
                        elif test_data[bayesian_network[j]].iloc[i] in data_freq[m][bayesian_network[j]]:
                            num = 1
                            den = train_data[(train_data[bayesian_network[j]]== test_data[bayesian_network[j]].iloc[i]) & (train_data['class']==m)].shape[0] + 1 +len(train_data[j].unique())
                        else:
                            num = 1
                            den = train_data[train_data['class']==m].shape[0] + 1 +len(train_data[j].unique())                            
                    prob_feature_given_parent_and_class = num /den
                likelihood[m] *= prob_feature_given_parent_and_class
        evidence = 0
        for k in attr_vals['class']:
            evidence += likelihood[k]*prob_of_class[k]            
        for n in attr_vals['class']:
            tanpredict[n] = likelihood[n]*prob_of_class[n]/evidence
        total+=1
        tanprediction += str (i+1) + " " + str(max(tanpredict, key=tanpredict.get)) + " " + str(test_data['class'].iloc[i]) + " " + str('{0:.12f}'.format(max(tanpredict.values()))) + "\n"
        if(max(tanpredict, key=tanpredict.get) == str(test_data['class'].iloc[i])):
            tan_count += 1
    print(tan_count)
    tan_accuracy.append(tan_count/total)
    return tanprediction            
            
    
    

def tan(train_data,test_data):
    attr_names,attr_vals = summarize(train_data)
    bayesian_network = prim(attr_names,attr_vals,train_data)
    print(bayesian_network)
    tanpredictions = tan_predict(attr_names,attr_vals,train_data,test_data,bayesian_network) 
    print(tanpredictions)

def naive(train_data,test_data):
    attr_names,attr_vals = summarize(train_data)
    prob_of_feature_value_given_class,prob_of_class = calculate_probs(train_data,attr_names,attr_vals)
    predictions = predict(test_data,prob_of_feature_value_given_class,prob_of_class,attr_names,attr_vals,train_data)
    print(predictions)

def comparison():
    data = readdata("chess-KingRookVKingPawn.arff")
    global naive_accuracy
    global tan_accuracy
    data=data.sample(frac=1).reset_index(drop=True)
    attr_names,attr_vals = summarize(data)
    data_first = data[data['class']==attr_vals['class'][0]]
    data_second = data[data['class']==attr_vals['class'][1]]
    test_data_size_first = data_first.shape[0]/10
    test_data_size_second = data_second.shape[0]/10
    for i in np.arange(0,10):
        
        if ((i+1)*test_data_size_first) <= data_first.shape[0]: 
            test_data_first = data_first[int(i*test_data_size_first):int((i+1)*test_data_size_first)]
            if i !=0:                
                train_data_first = data_first[0:int(i*test_data_size_first)]
                train_data_first.append(data_first[int(((i+1)*test_data_size_first)+1):])
            else:
                train_data_first = data_first[int(((i+1)*test_data_size_first)+1):]
        else:
            test_data_first = data_first[int(i*test_data_size_first):]
            if i!=0:
                train_data_first = data_first[0:int(i*test_data_size_first)]
            
        if ((i+1)*test_data_size_second) <= data_second.shape[0]: 
            test_data_second = data_second[int(i*test_data_size_second):int((i+1)*test_data_size_second)]
            if i !=0:
                train_data_second = data_second[0:int(i*test_data_size_second)]
                train_data_second.append(data_second[int(((i+1)*test_data_size_second)+1):])
            else:
                train_data_second = data_second[int(((i+1)*test_data_size_second)+1):]
        else:
            test_data_second = data_second[int(i*test_data_size_second):]
            if i !=0:
                train_data_second = data_second[0:int(i*test_data_size_second)]        

        test_datas = [test_data_first,test_data_second]
        test_data = pd.concat(test_datas)
        train_datas = [train_data_first,train_data_second]
        train_data = pd.concat(train_datas)
        train_data=train_data.sample(frac=1).reset_index(drop=True)
        test_data=test_data.sample(frac=1).reset_index(drop=True)
        naive(train_data,test_data)
        tan(train_data,test_data)
    print(naive_accuracy)
    print(tan_accuracy)
   

def main():
#    train_data = str(sys.argv[1])
#    test_data = str(sys.argv[2])
#    n = str(sys.argv[3])
    global naive_accuracy
    naive_accuracy = []
    global tan_accuracy
    tan_accuracy = []
    comparison()
    n = "k"
    train_data = "lymph_train.arff"
    test_data =  "lymph_test.arff"
    
    data_train = readdata(train_data)
    data_test = readdata(test_data)
    if n == "n":
        naive(data_train,data_test)
    elif n=="t":
        tan(data_train,data_test)
    else:
        print("Invalid option for model selection")
    
    
main()