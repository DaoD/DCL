import argparse
import random
import numpy as np
import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from BertSessionSearch import BertSessionSearch
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from Trec_Metrics import Metrics
from file_dataset import FileDataset
from MyDataLoaderNeg import Data
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--is_training",
                    default=True,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--per_gpu_batch_size",
                    default=20,
                    type=int,
                    help="The batch size.")
parser.add_argument("--per_gpu_test_batch_size",
                    default=256,
                    type=int,
                    help="The test batch size.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--pos_start_ratio",
                    default=0.33,
                    type=float,
                    help="")
parser.add_argument("--pos_ratio",
                    default=0.5,
                    type=float,
                    help="")
parser.add_argument("--neg_final_ratio",
                    default=0.5,
                    type=float,
                    help="")
parser.add_argument("--neg_ratio",
                    default=0.5,
                    type=float,
                    help="")
parser.add_argument("--task",
                    default="aol",
                    type=str,
                    help="Task")
parser.add_argument("--epochs",
                    default=6,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="./output/",
                    type=str,
                    help="The path to save score file.")
parser.add_argument("--pretrain_model_path",
                    default="",
                    type=str,
                    help="The path of pretrained model (COCA).")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--bert_model_path",
                    default="/",
                    type=str,
                    help="The path of BERT model.")
args = parser.parse_args()

if args.task == "aol":
    args.per_gpu_batch_size = 20
    args.pos_start_ratio = 0.3
    args.pos_ratio = 0.4
    args.neg_final_ratio = 0.7
    args.neg_ratio = 0.5
elif args.task == "tiangong":
    args.per_gpu_batch_size = 10
    args.pos_start_ratio = 0.5
    args.pos_ratio = 0.7
    args.neg_final_ratio = 0.5
    args.neg_ratio = 0.5
else:
    assert False

args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
args.save_path += BertSessionSearch.__name__ +  args.task + "." + str(int(args.pos_start_ratio * 10)) + "." + str(int(args.pos_ratio * 10)) + "." + str(int(args.neg_final_ratio * 10)) + "." + str(int(args.neg_ratio * 10))
args.log_path += BertSessionSearch.__name__ + args.task + "." + str(int(args.pos_start_ratio * 10)) + "." + str(int(args.pos_ratio * 10)) + "." + str(int(args.neg_final_ratio * 10)) + "." + str(int(args.neg_ratio * 10)) + ".log"
args.score_file_path += args.task + "/" + BertSessionSearch.__name__ + str(int(args.pos_start_ratio * 10)) + "." + str(int(args.pos_ratio * 10)) + "." + str(int(args.neg_final_ratio * 10)) + "." + str(int(args.neg_ratio * 10)) + ".score.txt"

logger = open(args.log_path, "a")
device = torch.device("cuda:0")
print(args)
logger.write("\nHyper-parameters:\n")
args_dict = vars(args)
for k, v in args_dict.items():
    logger.write(str(k) + "\t" + str(v) + "\n")

if args.task == "aol":
    train_data = "./data/aol/train.pos.id.sorted"
    train_ctx_dict = "./data/aol/train.ctx.id.txt"
    train_doc_dict = "./data/aol/train.doc.id.txt"
    train_ctx_vec = "./data/aol/train.ctx.vec.pt"
    train_doc_vec = "./data/aol/train.doc.vec.pt"
    train_candidate_doc = "./data/aol/train.doc.combined.pt"
    test_data = "./data/aol/valid_line.txt"
    predict_data = "./data/aol/test_line.txt"
    neg_num = 4
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 1
    tokenizer.add_tokens("[eos]")
    rerank = False
elif args.task == "tiangong":
    train_data = "./data/tiangong/train.pos.id.sorted"
    train_ctx_dict = "./data/tiangong/train.ctx.id.txt"
    train_doc_dict = "./data/tiangong/train.doc.id.txt"
    train_ctx_vec = "./data/tiangong/train.ctx.vec.pt"
    train_doc_vec = "./data/tiangong/train.doc.vec.pt"
    train_candidate_doc = "./data/tiangong/train.doc.combined.pt"
    test_last_data = "./data/tiangong/valid.txt"
    predict_data = "./data/tiangong/test.txt"
    neg_num = 9
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 2
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[empty_d]")
    rerank = False
else:
    assert False

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def pacing_function(x, t, c0):
    return ((x * ((1 - (c0 ** 2.0)) / t)) + (c0 ** 2.0)) ** 0.5

def pacing_functino_neg(x, t, c0):
    return 1.0 + c0 - (x * (1 - (c0 ** 2.0)) / t + (c0 ** 2.0)) ** 0.5

def hinge_loss(y_pred, margin=1.0):
    """
    Args:
        y_pred ([type]): [batch, N]
        y_label ([type]): [batch, N]
    """
    y_pred = y_pred.view(args.batch_size, -1)
    loss = torch.nn.functional.relu(margin - (torch.unsqueeze(y_pred[:, 0], -1) - y_pred[:, 1:]))
    return loss

def train_model():
    # load model
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)
    if args.pretrain_model_path != "":
        print("Load pretrained model...")
        model_state_dict = torch.load(args.pretrain_model_path)
        bert_model.load_state_dict({k.replace('bert_model.', ''):v for k, v in model_state_dict.items()}, strict=False)
    model = BertSessionSearch(bert_model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, test_data)

def train_step(model, train_data, bce_loss):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    y_pred = model.forward(train_data)
    loss = hinge_loss(y_pred)
    return loss

def fit(model, X_train, X_test, X_test_preq=None):
    train_dataset = Data(X_train, train_ctx_dict, train_doc_dict, train_ctx_vec, train_doc_vec, train_candidate_doc, rerank=rerank)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(train_dataset._sample_num * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.0), num_training_steps=t_total)
    one_epoch_step = train_dataset._sample_num // args.batch_size
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_result_pre = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    pos_data_fraction = args.pos_start_ratio
    pos_percentage_curriculum_iter = args.pos_ratio
    pos_curriculum_iterations = t_total * pos_percentage_curriculum_iter
    neg_data_fraction = 1.0
    neg_percentage_curriculum_iter = args.neg_ratio
    neg_curriculum_iterations = t_total * neg_percentage_curriculum_iter
    global_step = 0
    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        step = 0
        model.train()

        epoch_iterator = tqdm(range(int(t_total / args.epochs)), ncols=120)
        for step in epoch_iterator:
            training_data = train_dataset.get_train_next_batch(args.batch_size, pos_data_fraction, neg_data_fraction, neg_num=neg_num)
            loss = train_step(model, training_data, bce_loss)
            loss = loss.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=args.learning_rate, loss=loss.item())

            if step > 0 and step % (one_epoch_step // 5) == 0:
                best_result, _ = evaluate(model, X_test, bce_loss, best_result)
                model.train()
            avg_loss += loss.item()
            
            pos_data_fraction = min(1.0, pacing_function(global_step, pos_curriculum_iterations, c0=args.pos_start_ratio))
            neg_data_fraction = max(args.neg_final_ratio, pacing_functino_neg(global_step, neg_curriculum_iterations, c0=args.neg_final_ratio))

        cnt = train_dataset._sample_num // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        best_result, _ = evaluate(model, X_test, bce_loss, best_result)

def evaluate(model, X_test, bce_loss, best_result, is_test=False):
    y_pred, y_label = predict(model, X_test)
    metrics = Metrics(args.score_file_path, segment=50)

    with open(args.score_file_path, 'w') as output:
        for score, label in zip(y_pred, y_label):
            output.write(str(score) + '\t' + str(label) + '\n')
            
    result = metrics.evaluate_all_metrics()

    if not is_test and sum(result[2:]) > sum(best_result[2:]):
        # tqdm.write("save model!!!")
        best_result = result
        tqdm.write("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.write("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f \n" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.flush()
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    
    if is_test:
        tqdm.write("Best Result on Test: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (result[0], result[1], result[2], result[3], result[4], result[5]))
        logger.write("Best Result on Test: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f \n" % (result[0], result[1], result[2], result[3], result[4], result[5]))
    
    return best_result, best_result_pre

def predict(model, X_test, X_test_pre=None):
    model.eval()
    test_dataset = FileDataset(X_test, 128, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data, is_test=True)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["labels"].data.cpu().numpy().reshape(-1)
            y_label.append(y_tmp_label)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()

    return y_pred, y_label

def test_model():
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    model = BertSessionSearch(bert_model)
    model.bert_model.resize_token_embeddings(model.bert_model.config.vocab_size + additional_tokens)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    evaluate(model, predict_data, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], is_test=True)

if __name__ == '__main__':
    set_seed(0)
    if args.is_training:
        train_model()
        test_model()
    else:
        test_model()
    logger.close()
