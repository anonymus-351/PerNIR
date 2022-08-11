import numpy as np
from models.tifuknn import TIFUKNN
from models.global_oracle import GlobalOracle
from models.local_oracle import LocalOracle
from models.sknn import SKNN
from models.vsknn import VSKNN
from models.collab import Collab
from models.ppop import PPop
import pandas as pd
from metrics import *
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Mozhdeh')
    parser.add_argument('-user_index', type=int, default=0)
    args = parser.parse_args()
    return args


args = parse_args()

train_basket_path = 'data/instacart_30k/train_baskets.csv'
test_sample_path = 'data/instacart_30k/test_samples.csv'

train_baskets = pd.read_csv(train_basket_path)
test_samples = pd.read_csv(test_sample_path)

model = Collab(train_baskets,test_samples,args.user_index)
print('train...')
model.train()
print('predict...')
predicted_single_labels = model.predict()
result_df = pd.DataFrame(columns=['predicted_single_labels'])
result_df['predicted_single_labels'] = predicted_single_labels
result_df.to_csv('collab_instacart.csv',index=False)



result_df = pd.read_csv('collab_instacart.csv')
predicted_single_labels = result_df['predicted_labels']#.apply(eval).tolist()
test_label = test_samples['label_item'].tolist()
test_labels = test_samples['label_items']#.apply(eval).tolist()
test_inputs = test_samples['input_items'].tolist()
k_set = [10,20]
hr = []
p = []
r = []
mrr = []
ndcg = []
for k in k_set:
    print('k:',k)
    hr = []
    mrr = []
    for i in range(len(test_label)):
        if i%100000 == 0:
            print(i)
        label = int(test_label[i]) ######
        test_input = eval(test_inputs[i])
        #if len(test_input) < 30:
        #    continue
        labels = eval(test_labels[i])
        _pred = eval(predicted_single_labels[i])
        #_preds = eval(predicted_labels[i])
        #pred = predicted_single_labels[i]
        #preds = predicted_labels[i]
        pred = []
        #preds = []
        for item in _pred:
            if item not in test_input:
                pred.append(item)
        preds = pred
        if k == 'B':
            k = len(labels)
        hr.append(hr_k(label,pred,k))
        mrr.append(mrr_k(label,pred,k))
  #      p.append(precision_k(labels,preds,k))
   #     r.append(recall_k(labels,preds,k))
    #    ndcg.append(ndcg_k(labels,preds,k))
    print('hr:',np.mean(hr),len(hr))
    print('mrr:',np.mean(mrr))
#    print('recall:',np.mean(r))
#    print('precision:',np.mean(p))
 #   print('ndcg:',np.mean(ndcg))


