import torch.nn as nn
import torch.nn.init as init

class BertSessionSearch(nn.Module):
    def __init__(self, bert_model):
        super(BertSessionSearch, self).__init__()
        self.bert_model = bert_model
        self.classifier = nn.Linear(768, 1)
        init.xavier_normal_(self.classifier.weight)
    
    def forward(self, batch_data, is_test=False):
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        token_type_ids = batch_data["token_type_ids"]
        if not is_test:
            batch_size, num_docs, doc_len = batch_data["input_ids"].size()
            input_ids = input_ids.reshape(batch_size * num_docs, doc_len)
            attention_mask = attention_mask.reshape(batch_size * num_docs, doc_len)
            token_type_ids = token_type_ids.reshape(batch_size * num_docs, doc_len)
        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        sent_rep = self.bert_model(**bert_inputs)[1]
        y_pred = self.classifier(sent_rep)

        return y_pred.squeeze(1)