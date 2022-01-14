import torch
from tqdm import tqdm
import numpy as np
import random

class Data:
    def __init__(self, samples_file, context_dict, document_dict, context_rep, document_rep, candidate_index, max_seq_length=128, max_doc_length=40, rerank=False):
        self._max_seq_len = max_seq_length
        self._max_doc_len = max_doc_length
        self._all_train_samples = []
        self._all_train_ctx_dict = {}
        self._all_train_doc_dict = {}
        self._rerank = rerank
        with open(samples_file, "r") as fr:
            for line in fr:
                line = line.strip().split()
                self._all_train_samples.append((int(line[0]), int(line[1])))
        with open(context_dict, "r") as fr:
            for line in fr:
                line = line.strip().split("[=====]")
                self._all_train_ctx_dict[int(line[0])] = [int(x) for x in line[1].split()]
        with open(document_dict, "r") as fr:
            for line in fr:
                line = line.strip().split("[=====]")
                self._all_train_doc_dict[int(line[0])] = [int(x) for x in line[1].split()]
        self._all_ctx_rep = torch.load(context_rep)
        self._all_doc_rep = torch.load(document_rep)
        self._all_ctx_candidate_idx = torch.load(candidate_index)
        self._sample_num = len(self._all_train_samples)
        self._all_train_ids = list(range(self._sample_num))
        
    def annotate(self, context_ids, doc_ids):
        doc_ids = doc_ids[:self._max_doc_len]
        sample_ids = context_ids + [102] + doc_ids + [102]
        token_type_ids = [0] * (len(context_ids) + 1) + [1] * (len(doc_ids) + 1)
        sample_ids = sample_ids[-(self._max_seq_len - 1):]
        token_type_ids = token_type_ids[-(self._max_seq_len - 1):]
        sample_ids = [101] + sample_ids
        token_type_ids = [0] + token_type_ids
        attention_mask = [1] * len(sample_ids)
        assert len(sample_ids) <= self._max_seq_len
        while len(sample_ids) < self._max_seq_len:
            sample_ids.append(0)
            token_type_ids.append(0)
            attention_mask.append(0)
        assert len(sample_ids) == len(token_type_ids) == len(attention_mask) == self._max_seq_len
        return sample_ids, token_type_ids, attention_mask
    
    def select_doc(self, ctx_vec, pos_doc_vec, random_doc_vec, random_doc_idx, top_k=4, paccing_value=1.0):
        """
        Args:
            ctx_vec ([type]): [batch_size, hidden]
            pos_doc_vec ([type]): [batch_size, hidden]
            random_doc_vec ([type]): [batch_size, candidate_num, hidden]
            random_doc_idx ([type]): [batch_size, candidate_num]
            top_k ([type]): int
            cutoff_threshold (float, optional): float. Defaults to 0.8.
        """
        context_vec = np.asarray(ctx_vec)
        pos_doc_vec = np.asarray(pos_doc_vec)
        random_doc_vec = np.asarray(random_doc_vec)
        bsz = random_doc_vec.shape[0]
        candidate_scores = np.einsum("bd,bkd->bk", context_vec, random_doc_vec)  # [batch_size, candidate_num]
        result_index = []
        for k in range(bsz):
            one_scores = candidate_scores[k]
            one_candi_idx = random_doc_idx[k]
            candidate_idx = []
            for idx, s in enumerate(one_scores):
                candidate_idx.append((idx, s))
            if self._rerank:
                sorted_scores = sorted(candidate_idx, key=lambda x: x[1], reverse=True)
            else:
                sorted_scores = candidate_idx
            selected_idx = sorted_scores[:int(paccing_value * len(sorted_scores))]
            selected_idx = [x[0] for x in selected_idx]
            selected_idx = random.sample(list(selected_idx), top_k)
            one_res = []
            # one_res = selected_idx
            for one_idx in selected_idx:
                one_res.append(one_candi_idx[one_idx])
            result_index.append(one_res)
        return result_index

    def select_top_k_docs(self, ctx_idx_list, pos_doc_idx_list, random_docs_list, top_k=4, paccing_value=1.0):
        """
        Args:
            ctx_idx_list ([type]): [batch_size]
            pos_doc_idx_list ([type]): [batch_size]
            random_docs_list ([type]): [batch_size, candidate_num]
            top_k ([type]): int
        """
        batch_ctx_vec = [self._all_ctx_rep[one_id] for one_id in ctx_idx_list]
        batch_pos_doc_vec = [self._all_doc_rep[one_id] for one_id in pos_doc_idx_list]
        batch_candi_doc_vec = []
        for idx_list in random_docs_list:
            random_doc_rep = [self._all_doc_rep[one_id] for one_id in idx_list]
            batch_candi_doc_vec.append(random_doc_rep)
        batch_select_response_index = self.select_doc(batch_ctx_vec, batch_pos_doc_vec, batch_candi_doc_vec, random_docs_list, top_k=top_k, paccing_value=paccing_value)
        assert np.array(batch_select_response_index).shape == (len(ctx_idx_list), top_k)
        return batch_select_response_index

    def get_train_next_batch(self, batch_size, pos_pacing_value, neg_pacing_value, neg_num=4):
        pacing_num = int(self._sample_num * pos_pacing_value)
        train_ids = self._all_train_ids[:pacing_num]
        batch_idx_list = random.sample(train_ids, batch_size)

        all_batch_input_ids = []
        all_batch_token_type_ids = []
        all_batch_attention_masks = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_masks = []
        batch_random_docs_list = []
        batch_labels = []
        batch_ctx_idx_list = []
        batch_pos_doc_idx_list = []
        for idx in batch_idx_list:
            context_idx, pos_doc_idx = self._all_train_samples[idx]
            context_ids = self._all_train_ctx_dict[context_idx]
            pos_doc_ids = self._all_train_doc_dict[pos_doc_idx]
            cancidate_idx_list = self._all_ctx_candidate_idx[idx]
            assert pos_doc_idx not in cancidate_idx_list
            batch_random_docs_list.append(cancidate_idx_list)
            sample_ids, token_type_ids, attention_mask = self.annotate(context_ids, pos_doc_ids)
            batch_input_ids.append(sample_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_ctx_idx_list.append(context_idx)
            batch_pos_doc_idx_list.append(pos_doc_idx)
        all_batch_input_ids.append(batch_input_ids)
        all_batch_token_type_ids.append(batch_token_type_ids)
        all_batch_attention_masks.append(batch_attention_masks)
        batch_sample_doc_idx_list = self.select_top_k_docs(batch_ctx_idx_list, batch_pos_doc_idx_list, batch_random_docs_list, top_k=neg_num, paccing_value=neg_pacing_value)
        one_neg_doc_idx = batch_sample_doc_idx_list[0][0]
        batch_labels.append([1] * batch_size)
        for i in range(neg_num):
            one_batch_neg_doc_id, one_batch_token_type_id, one_batch_attention_mask = [], [], []
            batch_labels.append([0] * batch_size)
            for j in range(batch_size):
                one_neg_doc_idx = batch_sample_doc_idx_list[j][i]
                one_neg_doc_ids = self._all_train_doc_dict[one_neg_doc_idx]
                context_idx = batch_ctx_idx_list[j]
                context_ids = self._all_train_ctx_dict[context_idx]
                sample_ids, token_type_ids, attention_mask = self.annotate(context_ids, one_neg_doc_ids)
                one_batch_neg_doc_id.append(sample_ids)
                one_batch_token_type_id.append(token_type_ids)
                one_batch_attention_mask.append(attention_mask)
            all_batch_input_ids.append(one_batch_neg_doc_id)
            all_batch_token_type_ids.append(one_batch_token_type_id)
            all_batch_attention_masks.append(one_batch_attention_mask)
        batch = {
            'input_ids': torch.LongTensor(all_batch_input_ids).permute(1, 0, 2), 
            'token_type_ids': torch.LongTensor(all_batch_token_type_ids).permute(1, 0, 2), 
            'attention_mask': torch.LongTensor(all_batch_attention_masks).permute(1, 0, 2), 
            'labels': torch.FloatTensor(batch_labels).permute(1, 0)
        }
        return batch