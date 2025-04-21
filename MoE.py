import torch
from torch import nn
import torch.nn.functional as F


def switch_load_balancing_loss(router_logits, expert_num, top_k):
    router_probs = F.softmax(router_logits, dim=-1)
    # (b*seq_len, expert_num)
    
    _, selected_expert = torch.topk(router_probs, k=top_k, dim=-1)
    # (b*seq_len, top_k)
    expert_mask = F.one_hot(selected_expert, expert_num).to(torch.float)
    # (b*seq_len, top_k, expert_num)
    actual_load = expert_mask.mean(dim=0)
    # (top_k, expert_num)

    aux_loss = torch.sum(actual_load * router_probs.mean(dim=0)) * expert_num

    z_loss = torch.mean(router_logits**2)
    z_loss_weight = 0.01

    return aux_loss + z_loss * z_loss_weight


class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.fc = nn.Linear(feature_in, feature_out)
    
    def forward(self, x):
        return self.fc(x)


class MOEConfig:
    def __init__(self, in_dim, out_dim, expert_num, top_k=2, shared_expert_num=1):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expert_num = expert_num
        self.top_k = top_k
        self.shared_expert_num = shared_expert_num


class MOERouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(config.in_dim, config.in_dim//2),
            nn.Linear(config.in_dim//2, config.expert_num)
            )
        self.top_k = config.top_k
        self.expert_num = config.expert_num
    
    def forward(self, x):
        router_logits = self.gate(x)
        # router_logits shape (b*seq_len, expert_num)
        router_probs = F.softmax(router_logits, dim=-1)

        router_weights, top_k_indices = router_probs.topk(k=self.top_k, dim=-1)
        # router_weight & top_k_indices shape (b*seq_len, top_k)

        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        expert_masks = F.one_hot(top_k_indices, self.expert_num)
        # expert_mask shape (b*seq_len, top_k, expert_num)
        expert_masks = expert_masks.permute(2, 1, 0)
        # expert_mask shape (expert_num, top_k, b*seq_len)
        
        return router_logits, router_weights, top_k_indices, expert_masks


class SparseMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.in_dim = config.in_dim
        self.out_dim = config.out_dim
        self.expert_num = config.expert_num

        self.experts = nn.ModuleList([
            BasicExpert(
                config.in_dim,
                config.out_dim
            ) for _ in range(config.expert_num)
        ])
        self.router = MOERouter(config)

    def forward(self, x):
        # x shape (b, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape

        # reshape to (b*seq_len, hidden_dim) for token level calculation
        x = x.reshape(-1, hidden_dim)

        router_logtis, router_weights, top_k_indices, expert_masks = self.router(x)

        final_x = torch.zeros((batch_size*seq_len, self.config.out_dim), dtype=x.dtype, device=x.device)

        for expert_idx in range(self.expert_num):
            expert_layer = self.experts[expert_idx]

            current_expert_mask = expert_masks[expert_idx]
            # current_expert_mask shape (top_k, b*seq_len)
            top_idx, token_idx = torch.where(current_expert_mask)
            # top_idx shape: (top_k), token_idx shape: (b*seq_len)

            current_x = x[token_idx]
            # current_x shape (selected_token_num, hidden_dim)
            current_x = expert_layer(current_x)
            current_token_router_weight = router_weights[token_idx, top_idx]
            # current_token_router_weight shape (selected_token_num)
            current_token_router_weight = current_token_router_weight.unsqueeze(-1)
            # current_token_router_weight shape (selected_token_num, 1)

            current_x *= current_token_router_weight

            final_x.index_add_(0, token_idx, current_x)

        final_x = final_x.reshape(batch_size, seq_len, self.config.out_dim)

        return final_x, router_logtis


class SharedExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.routed_experts = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(self.config.in_dim, self.config.out_dim)
            ]
        )
        self.router_logits = None

    def forward(self, x):
        # x shape: (b, seq_len, hidden_dim)
        b, seq_len, hidden_dim = x.shape

        shared_expert_output_list = [expert(x) for expert in self.shared_experts]
        shared_expert_output = torch.stack(shared_expert_output_list, dim=0)
        # shared_expert_output shape (shared_expert_num, b, seq_len, hidden_dim)

        shared_expert_output = shared_expert_output.sum(dim=0)
        # shared_expert_output shape (b, seq_len, hidden_dim)
        spared_moe_out, router_logits = self.routed_experts(x)

        self.router_logits = router_logits

        output = spared_moe_out + shared_expert_output

        return output
    
def switch_load_balancing_loss(router_logits, expert_num, top_k):
    router_probs = F.softmax(router_logits, dim=-1)
    # (b*seq_len, expert_num)
    _, selected_expert = torch.topk(router_probs, k=top_k, dim=-1)
    # (b*seq_len, top_k)
    expert_mask = F.one_hot(selected_expert, expert_num).to(torch.float)
    # (b*seq_len, top_k, expert_num)
    actual_load = expert_mask.mean(dim=0)
    # (top_k, expert_num)

    aux_loss = torch.sum(actual_load * router_probs.mean(dim=0)) * expert_num

    z_loss = torch.mean(router_logits**2)
    z_loss_weight = 0.01

    return aux_loss + z_loss * z_loss_weight