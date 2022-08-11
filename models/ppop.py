import numpy as np


class PPop:
    def __init__(self,train_baskets,test_samples):
        self.train_baskets = train_baskets
        self.test_samples = test_samples

    def train(self):
        pass

    def predict(self):
        test_inputs = self.test_samples['input_items'].apply(eval).tolist()
        test_users = self.test_samples['user_id'].tolist()

        train_user_item = self.train_baskets.groupby(['user_id','item_id']).size().to_frame(name = 'item_count'). \
            reset_index().sort_values(['user_id','item_count'],ascending=False)
        personal_topk_df = train_user_item.groupby('user_id')['item_id'].apply(list).reset_index()
        personal_topk_dict = dict(zip(personal_topk_df['user_id'],personal_topk_df['item_id']))

        predictions = []
        for i,input_items in enumerate(test_inputs):
            if i%100000 == 0:
                print(i)
            user = test_users[i]
            candids = personal_topk_dict[user]
            preds = []
            for item in candids:
                if item not in input_items:
                    preds.append(item)
            predictions.append(preds)
        return predictions
