from transformers import BertTokenizer, BertForSequenceClassification
from utils import read_csv_tsv_expert, load_aligner, load_text
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from clip_aligner import CLIPAligner
import json
import h5py
import numpy as np
import torch.nn.functional as F
from MoE import SharedExpertMOE, MOEConfig, switch_load_balancing_loss


def load_img(img_path_list):
    img_dir_list = []
    for img_path in img_path_list:
        with open(img_path, "r") as f:
            idx2embedding = json.load(f)
            embedding_list = [idx2embedding[k] for k in sorted(idx2embedding, key=int)]
            img_dir_list.extend(embedding_list)
    return img_dir_list


def load_speech(speech_path_list):
    speech_dir_list = []
    for speech_path in speech_path_list:
        with h5py.File(speech_path, "r") as f:
            idx2embedding = {}
            for key in f.keys():
                idx2embedding[key] = f[key][()]
            embedding_list = [idx2embedding[k] for k in sorted(idx2embedding, key=int)]
            speech_dir_list.extend(embedding_list)
    return speech_dir_list


def load_expert_weight(model, args, index):
    for weight_path in Path(args.expert_weight_dir).rglob("*.pt"):
        if "image" in weight_path.name:    
            saved_image_expert_weight = torch.load(weight_path)
        elif "speech" in weight_path.name:
            saved_speech_expert_weight = torch.load(weight_path)
        elif "text" in weight_path.name:
            saved_text_expert_weight = torch.load(weight_path)
    model.shared_experts[0].fc.load_state_dict(saved_text_expert_weight[f"layer_{index}_intermediate_dense"])
    model.routed_experts.experts[0].fc.load_state_dict(saved_speech_expert_weight[f"layer_{index}_intermediate_dense"])
    model.routed_experts.experts[1].fc.load_state_dict(saved_image_expert_weight[f"layer_{index}_intermediate_dense"])


class MMDataset(Dataset):
    def __init__(self, text_paths, img_paths, speech_paths):
        super().__init__()
        self.text_list = load_text(text_paths)
        self.img_list = load_img(img_paths)
        self.speech_list = load_speech(speech_paths)
        self.labels = self.get_labels(text_paths)

    def get_labels(self, paths):
        labels = []
        for path in paths:
            _, label = read_csv_tsv_expert(path)
            labels.extend(label)
        return labels
 
    def __getitem__(self, index):
        return (self.text_list[index], self.speech_list[index], self.img_list[index]), self.labels[index]
    
    def __len__(self):
        return len(self.labels)

def mm_collate_fn(batch, args):
    texts = [item[0][0] for item in batch]
    tokenizer = BertTokenizer.from_pretrained(args.bert_base_model)
    texts = tokenizer(texts, padding=True, return_tensors="pt")["input_ids"].to(args.device)

    imgs = [item[0][2] for item in batch]
    seq_len = max([len(img_embd) for img_embd in imgs])
    imgs = torch.stack([F.pad(torch.tensor(item[0][2]), (0, 0, 0, seq_len - np.array(item[0][2]).shape[-2])) for item in batch], dim=0).to(args.device)
    
    speechs = [torch.tensor(item[0][1]).squeeze() for item in batch]
    speechs = F.interpolate(torch.stack(speechs, dim=0).permute(0, 2, 1), size=100, mode='linear', align_corners=False)
    speechs = speechs.permute(0, 2, 1).to(args.device).float()

    labels = [item[1] for item in batch]
    labels = torch.Tensor(labels).to(args.device).long()

    return (texts, speechs, imgs), labels


class MoEBert(nn.Module):
    def __init__(self, args, config, moe_position):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(args.bert_base_model, num_labels=2)
        self.replace_FNN_to_MoE(args, config, moe_position)

    def replace_FNN_to_MoE(self, args, config, moe_position):
        moe_list = nn.ModuleList([SharedExpertMOE(config) for _ in range(len(moe_position))])
        n = 0
        for i, layer in enumerate(self.bert.bert.encoder.layer):
            if i in moe_position:
                layer.intermediate.dense = moe_list[n]
                load_expert_weight(layer.intermediate.dense, args, i)
                n += 1
    
    def forward(self, x):
        x = self.bert.bert.encoder(x)
        x = self.bert.bert.pooler(x["last_hidden_state"])
        x = self.bert.dropout(x)
        x = self.bert.classifier(x)
        return x


class MMBert(nn.Module):
    def __init__(self, args, moe_config, moe_position):
        super().__init__()
        self.img_aligner = CLIPAligner(in_feature=512, out_feature=768)
        img_aligner_weight_path = self.get_aligner_weight_path(args.aligner_weight_dir, "image")
        load_aligner(self.img_aligner, img_aligner_weight_path)

        self.speech_aligner = CLIPAligner(in_feature=512, out_feature=768)
        speech_aligner_weight_path = self.get_aligner_weight_path(args.aligner_weight_dir, "speech")
        load_aligner(self.speech_aligner, speech_aligner_weight_path)

        self.moe_bert = MoEBert(args, moe_config, moe_position)

    def get_aligner_weight_path(self, aligner_weight_dir, modality):
        aligner_weight_dir = Path(aligner_weight_dir)
        for weight_path in aligner_weight_dir.rglob("*pth"):
            if modality in weight_path.name:
                return weight_path
            
    def get_router_logits(self):
        router_logit_list = []
        for layer in self.moe_bert.bert.bert.encoder.layer:
            router_logit_list.append(layer.intermediate.dense.router_logits)
        return router_logit_list
            
    def forward(self, text_ids, speech_embd, img_embd):
        img_aligned = self.img_aligner(img_embd)
        speech_aligned = self.speech_aligner(speech_embd)
        text_embd = self.moe_bert.bert.bert.embeddings(text_ids)
        concat_embd = torch.concatenate([text_embd, speech_aligned, img_aligned], dim=1)
        logits = self.moe_bert(concat_embd)
        return logits


def train_step(model, optimizer, loss_fn, dataloader, args):
    loss_val = 0
    model.train()
    for x, y in dataloader:
        texts, speechs, imgs = x
        pred = model(texts, speechs, imgs)
        router_logits_list = model.get_router_logits()
        router_losses = [
            switch_load_balancing_loss(router_logits, args.routed_expert_num, args.topk_expert)
            for router_logits in router_logits_list
        ]
        loss = args.loss_fn_factor * loss_fn(pred, y) + (1-args.loss_fn_factor) * torch.stack(router_losses).sum() / len(router_losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
    loss_val /= len(dataloader)
    return loss_val, 


def val_step(model, dataloader, args):
    acc = 0
    model.eval()
    with torch.inference_mode():
        for x, y in dataloader:
            texts, speechs, imgs = x
            pred = model(texts, speechs, imgs)
            _, pred = pred.max(dim=-1)
            acc += sum(pred==y).item() / len(pred)
    acc /= len(dataloader)        
    return acc


def train(model, optimizer, loss_fn, train_dataloader, val_dataloader, args):
    loss_list = []
    acc_list = []
    for epoch in tqdm(range(args.epoch_num)):
        loss_val = train_step(model, optimizer, loss_fn, train_dataloader, args)
        acc_val = val_step(model, val_dataloader, args)
        print(f"Epoch{epoch+1} | loss:{loss_val} | acc:{acc_val}")
        loss_list.append(loss_val)
        acc_list.append(acc_val)
    return {"train_loss": loss_list, "val_acc": acc_list}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_base_model", default="bert-base-chinese", type=str)
    parser.add_argument("--batch_size", default=64,  type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",  type=str)
    parser.add_argument("--epoch_num", default=50, type=int)
    parser.add_argument("--lr", default=0.5e-4,  type=float) 
    parser.add_argument("--aligner_weight_dir", default="aligner_weight", type=str)
    parser.add_argument("--expert_weight_dir", default="expert_weight", type=str)
    parser.add_argument("--train_log_dir", default="train_log", type=str)
    parser.add_argument("--routed_expert_num", default=2, type=int)
    parser.add_argument("--shared_expert_num", default=1, type=int)
    parser.add_argument("--topk_expert", default=2, type=int)
    parser.add_argument("--loss_fn_factor", default=0.9, type=float)
    args = parser.parse_args()

    save_dir = Path(args.expert_weight_dir)
    moe_config = MOEConfig(in_dim=768, out_dim=3072, expert_num=args.routed_expert_num, shared_expert_num=args.shared_expert_num, top_k=args.topk_expert)
    moe_position = list(range(12))
    model = MMBert(args, moe_config, moe_position).to(args.device)
    for param in model.moe_bert.parameters():
        param.requires_grad = False
    for layer in model.moe_bert.bert.bert.encoder.layer:
        for param in layer.intermediate.parameters():
            param.requires_grad = True
    for param in model.moe_bert.bert.classifier.parameters():
        param.requires_grad = True
    print(model)
    for n, p in model.named_parameters():
        print(n, p.requires_grad)
    text_list = [
        "ToxiCloakCN/Datasets/base_data.tsv", 
        "ToxiCloakCN/Datasets/Only_Keywords/emoji_keyword.csv", 
        "ToxiCloakCN/Datasets/Only_Keywords/homo_keyword.csv"
    ]
    img_list = [
        "base_image_data_processed/base_image_data.json",
        "emoji_image_data_processed/emoji_image_data.json",
        "homo_image_data_processed/homo_image_data.json"
    ]
    speech_list = [
        "base_speech_data_processed/base_speech_data.h5",
        "emoji_speech_data_processed/emoji_speech_data.h5",
        "homo_speech_data_processed/homo_speech_data.h5"
    ]
    dataset = MMDataset(text_list, img_list, speech_list)
    train_size = int(len(dataset)*0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: mm_collate_fn(batch, args))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: mm_collate_fn(batch, args))
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    train_dict = train(model, optimizer, loss_fn, train_dataloader, val_dataloader, args)
    train_log_path = args.train_log_dir + "/moe_lr_" + str(args.lr) + ".json"
    with open(train_log_path, "w") as f:
        json.dump(train_dict, f)
    with open(train_log_path, "w") as f:
        json.dump(train_dict, f)
    weight_path = save_dir / "mmbert_weight.pt"
    torch.save(model.state_dict(), weight_path)

