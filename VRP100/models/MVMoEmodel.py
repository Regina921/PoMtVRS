import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
# from tutel import moe as tutel_moe
from utils.functions import batchify, gather_by_index, unbatchify, unbatchify_and_gather
from typing import Tuple, Union
from dataclasses import dataclass, fields
from torch import Tensor
from .MOELayer import MoE

__all__ = ['MOEModel']

 
def linear_layer(input_dim, output_dim, std=1e-2, bias=True):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim,bias=bias)
    nn.init.normal_(linear.weight, std=std)  
    nn.init.zeros_(linear.bias)  
    return linear


@dataclass
class PrecomputedCache:  
    node_embeddings: Tensor   
    glimpse_key: Tensor  
    glimpse_val: Tensor
    logit_key: Tensor  

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))  

    def batchify(self, num_starts):   
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)

 
class PromptNet(nn.Module):   
    def __init__(self, args):
        super().__init__()
        input_dim = 5  
        output_dim = args.model_params['embedding_dim']
        self.logit_clipping = args.model_params['logit_clipping']
        
        layer1 = nn.Linear(input_dim, output_dim,bias=False)  
        nn.init.uniform_(layer1.weight)  
        self.model = nn.Sequential(
            layer1,   # 5->E
            nn.LayerNorm(output_dim),   
            linear_layer(output_dim,output_dim),   
            nn.ReLU(),  
            linear_layer(output_dim, output_dim//8), # task embedding  
            nn.LayerNorm(output_dim//8),  
            linear_layer(output_dim//8, 5*output_dim),  
        )

    def forward(self, td):   
        return {"prompt": self.model(td['p_s_tag'][:,:5]).view(td.batch_size[0], 5, -1)}  # (B, 5, E)

 
class MVMOEModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_params = args.model_params
        self.eval_type = self.model_params['eval_type']
        # self.problem = self.model_params['problem']
        self.aux_loss = 0

        self.encoder = MTL_Encoder(**args.model_params)
        self.decoder = MTL_Decoder(**args.model_params)
        self.encoded_nodes = None  # (batch, problem+1, EMBEDDING_DIM)
        self.now_p_type = None
        self.prompt_net = PromptNet(args)  

    @staticmethod
    def greedy(logprobs, mask=None):  
        """Select the action with the highest probability."""
        # [BS], [BS]
        selected = logprobs.argmax(dim=-1)
        if mask is not None:
            assert (not (~mask).gather(1, selected.unsqueeze(-1)).data.any()), "infeasible action selected"
        return selected

    @staticmethod
    def sampling(logprobs, log, mask=None):  
        """Sample an action with a multinomial distribution given by the log probabilities."""
        probs = logprobs.exp()  
        selected = torch.multinomial(probs, 1).squeeze(1) #
        if mask is not None:
            while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
                log("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
            assert (not (~mask).gather(1, selected.unsqueeze(-1)).data.any()), "infeasible action selected"
        return selected

    def forward(self, td, env):
         
        args = self.args
        # init prompt
        p_out = self.prompt_net(td)
        prompt = p_out['prompt']
 
        # MOEencoder 
        node_embed, moe_loss = self.encoder(td, prompt) 
        self.aux_loss = moe_loss
 
        # multi_start node
        num_starts, action = env.select_start_nodes(td) # action repeat_num*bs   action num_starts*batch_size [1]*batch_size + [2]*batch_size + ...
        td = batchify(td, num_starts) # = td.expand(num_starts, batch_size).contiguous().view(batch_size * num_starts,)  td['locs'][0] == td['locs'][batch_size]
        logprobs_list = [torch.zeros_like(action, device=td.device)]
        actions_list = [action]
        # first node
        td.set("action", action)
        td = env.step(td)["next"]
        # set decoder k v
        # print(node_embed.shape)

        decoder_k = reshape_by_heads(self.decoder.Wk(node_embed), head_num=args.model_params['head_num'])
        decoder_v = reshape_by_heads(self.decoder.Wv(node_embed), head_num=args.model_params['head_num']) # (batch, head_num, problem+1, qkv_dim)v
        decoder_single_head_k = node_embed.transpose(1, 2) # (batch, embedding, problem+1)
        cache = PrecomputedCache(node_embed, decoder_k, decoder_v, decoder_single_head_k)
        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logprobs, mask, moe_loss = self.decoder(td, cache, num_starts)
            self.aux_loss += moe_loss

            # select
            if self.training:  # 
                select = MVMOEModel.sampling(logprobs, args.log, mask)
            else:
                select = MVMOEModel.greedy(logprobs,mask)
            logprobs = gather_by_index(logprobs, select, dim=1)
            td.set("action", select)
            actions_list.append(select)
            logprobs_list.append(logprobs)
            td = env.step(td)["next"]
            step += 1
        # post op
        logprobs = torch.stack(logprobs_list, 1)
        actions = torch.stack(actions_list, 1)
        td.set("reward", env.get_reward(td, actions))  
        assert (logprobs > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        log_likelihood_sum = logprobs.sum(1)  # Calculate log_likelihood [batch]
        outdict = {"reward": td["reward"], "log_likelihood": log_likelihood_sum,     # bs * repeat_num
        }
        return outdict


################################################################################
###[Encoder]
class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        hidden_dim = self.model_params['ff_hidden_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # self.p_num = self.model_params['p_num']  # 5

        # [Option 1]: Use MoEs in Raw Features
        if self.model_params['num_experts'] > 1 and "Raw" in self.model_params['expert_loc']:
            self.embedding_depot = MoE(input_size=3, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                       k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                       routing_method=self.model_params['routing_method'], moe_model="Linear")
            self.embedding_node = MoE(input_size=7, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                      k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                      routing_method=self.model_params['routing_method'], moe_model="Linear")
        else:
            # self.embedding_depot = nn.Linear(2, embedding_dim)
            # self.embedding_node = nn.Linear(5, embedding_dim)
            self.embedding_depot = nn.Linear(3, embedding_dim) # locs, distance_limit, time_windows # TODO:
            self.embedding_node = nn.Linear(7, embedding_dim)  # 
        self.layers = nn.ModuleList([EncoderLayer(i, **model_params) for i in range(encoder_layer_num)])

    def forward(self, td, prompt):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand_tw.shape: (batch, problem, 5)
        # prob_emb: (1, embedding)
        """
        :param depot_xy: (batch, 1, 2)
        :param node_xy_demand: (batch, problem, 3)
        :return: out # (batch, problem+1, embedding)   
        """
        # depot_feats = td["locs"][:, :1, :] # (batch, 1, 2)   
        depot_feats = torch.cat(   # [B,1,3]
            [
                td["locs"][:, :1, :],
                td["distance_limit"][..., None],
            ], -1, ) 
 
        node_feats = torch.cat(
            (
                td["demand_linehaul"][..., 1:, None],   
                td["demand_backhaul"][..., 1:, None],  
                td["time_windows"][..., 1:, :],   
                td["service_time"][..., 1:, None],   
                td["locs"][:, 1:, :],   
            ), -1,)    # (batch, N, 7)
        
        depot_feats = torch.nan_to_num(depot_feats, nan=0.0, posinf=0.0, neginf=0.0)   
        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
        bs, n, _7 = node_feats.shape   
        # global_embeddings = self.embedding_depot(depot_feats)  # [batch, 1, embed_dim]   (B,1,D)  
        # cust_embeddings = self.embedding_node(node_feats)  # [batch, N, embed_dim]
 
        moe_loss = 0
        if isinstance(self.embedding_depot, MoE) or isinstance(self.embedding_node, MoE):
            global_embeddings, loss_depot = self.embedding_depot(depot_feats)
            cust_embeddings, loss_node = self.embedding_node(node_feats)
            moe_loss = moe_loss + loss_depot + loss_node
        else:
 
            global_embeddings = self.embedding_depot(depot_feats)  # [batch, 1, embed_dim]  
            cust_embeddings = self.embedding_node(node_feats)  # [batch, N, embed_dim]

        out = torch.cat((global_embeddings, cust_embeddings), -2)  # [batch, N+1, embed_dim]
        # out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out, loss = layer(out)
            moe_loss = moe_loss + loss

        return out, moe_loss
        # shape: (batch, problem+1, embedding)
  
 
class EncoderLayer(nn.Module):
    def __init__(self, depth=0, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        # [Option 2]: Use MoEs in Encoder
        if self.model_params['num_experts'] > 1 and "Enc{}".format(depth) in self.model_params['expert_loc']:
            # TODO: enabling parallelism
            # (1) MOE with tutel, ref to "https://github.com/microsoft/tutel"
            """
            assert self.model_params['routing_level'] == "node", "Tutel only supports node-level routing!"
            self.feedForward = tutel_moe.moe_layer(
                gate_type={'type': 'top', 'k': self.model_params['topk']},
                model_dim=embedding_dim,
                experts={'type': 'ffn', 'count_per_node': self.model_params['num_experts'],
                         'hidden_size_per_expert': self.model_params['ff_hidden_dim'],
                         'activation_fn': lambda x: F.relu(x)},
            )
            """
            # (2) MOE with "https://github.com/davidmrau/mixture-of-experts"
            self.feedForward = MoE(input_size=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                   hidden_size=self.model_params['ff_hidden_dim'], k=self.model_params['topk'], T=1.0, noisy_gating=True,
                                   routing_level=self.model_params['routing_level'], routing_method=self.model_params['routing_method'], moe_model="MLP")
        else:
            self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        """
        Two implementations:
            norm_last: the original implementation of AM/POMO: MHA -> Add & Norm -> FFN/MOE -> Add & Norm
            norm_first: the convention in NLP: Norm -> MHA -> Add -> Norm -> FFN/MOE -> Add
        """
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num, moe_loss = self.model_params['head_num'], 0

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)  # (batch, problem, HEAD_NUM*KEY_DIM)
            multi_head_out = self.multi_head_combine(out_concat)  # (batch, problem, EMBEDDING_DIM)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2, moe_loss = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)  # (batch, problem, EMBEDDING_DIM)
        else:
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2, moe_loss = self.feedForward(out2)
            out3 = input2 + out2

        return out3, moe_loss

########################################
 
class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 5, head_num * qkv_dim, bias=False)  # 
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # [Option 3]: Use MoEs in Decoder
        if self.model_params['num_experts'] > 1 and 'Dec' in self.model_params['expert_loc']:   
            self.multi_head_combine = MoE(input_size=head_num * qkv_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                          k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                          routing_method=self.model_params['routing_method'], moe_model="Linear")
 
        else:
            self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

        # ++ 
        self.use_gate = self.model_params['use_gate']
        if self.model_params['use_gate']:
            self.W_gate = nn.Linear(embedding_dim + 5, head_num, bias=False)
             
        self.attr_mapping = nn.Linear(5, embedding_dim, bias=False)

        # self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)
        # self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)
 
    def gate_and_attention_block(self, out_concat, context_embedding, cur_node_embedding, state_embedding):

        B, S, HD = out_concat.shape
        H = self.model_params['head_num']
        D = self.model_params['qkv_dim']

        y = out_concat.view(B, S, H, D)   
        gate = torch.sigmoid(self.W_gate(context_embedding))  # [B, S, H]
        y = y * gate.unsqueeze(-1)        
        out_concat = y.view(B, S, H * D)   
        # 
        if isinstance(self.multi_head_combine, MoE):
            mh_atten_out, moe_loss = self.multi_head_combine(out_concat)
        else:
            mh_atten_out = self.multi_head_combine(out_concat)  # shape: (batch, pomo, embedding) 
  
        ##====[2]=============================
        mh_atten_out = mh_atten_out + cur_node_embedding + self.attr_mapping(state_embedding.clone())
        mh_atten_out = self.feedForward(mh_atten_out)[0] + mh_atten_out

        return mh_atten_out
 

    def forward(self, td, cache, num_starts):
        moe_loss = 0

        td = unbatchify(td, num_starts)    # num_starts * bs -> bs , num_starts   
        # get last node   
        cur_node_embedding = gather_by_index(cache.node_embeddings, td["current_node"])

        # get state original 
        remaining_linehaul_capacity = td["vehicle_capacity"] - td["used_capacity_linehaul"]
        remaining_backhaul_capacity = td["vehicle_capacity"] - td["used_capacity_backhaul"]
        # (B, num_starts, 5)
        state_embedding = torch.cat([ remaining_linehaul_capacity, remaining_backhaul_capacity, td["current_time"], td["current_route_length"], td["open_route"] ], -1)   # bs, num start, 5
        
        # input_cat 
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1) # bs,num_locs,embed+5
        # get q
        glimpse_q = reshape_by_heads(self.Wq_last(context_embedding), head_num=self.model_params['head_num'])  # (B, head_num, num_starts, qkv_dim)
        mask = td["action_mask"]
        # mha  
        out_concat = multi_head_attention(glimpse_q, cache.glimpse_key, cache.glimpse_val, rank3_ninf_mask=mask)    # (batch, pomo, head_num*qkv_dim)
  
        ##=================================================
        if self.use_gate:
            mh_atten_out = self.gate_and_attention_block(out_concat, context_embedding, cur_node_embedding, state_embedding)
        else: 
            # 
            if isinstance(self.multi_head_combine, MoE):
                mh_atten_out, moe_loss = self.multi_head_combine(out_concat)
            else:
                mh_atten_out = self.multi_head_combine(out_concat)  # shape: (batch, pomo, embedding) 
 
        ##===================================================
        # sha  
        score = torch.matmul(mh_atten_out, cache.logit_key) # (batch, pomo, problem)  
        score_scaled = score / self.model_params['sqrt_embedding_dim'] # (batch, pomo, problem) 
        # post op
        logits = rearrange(score_scaled, "b s l -> (s b) l", s=num_starts)  
        mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        logits = torch.tanh(logits) * self.model_params['logit_clipping']  
        logits[~mask] = float("-inf")
        # logits = logits / temperature  # temperature scaling
        # probs = F.softmax(score_masked, dim=2)# (batch, pomo, problem)

        log_probs = F.log_softmax(logits, dim=-1)   # (batch, pomo, problem)
        return log_probs, mask, moe_loss  # Compute log probabilities
 
 

########################################
# NN SUB CLASS / FUNCTIONS
def reshape_by_heads(qkv, head_num):
    # q.(batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1) # (batch, n, head_num, key_dim)
    q_transposed = q_reshaped.transpose(1, 2) # (batch, head_num, n, key_dim)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)
    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)
    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans

#  
class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)
    def forward(self, input1):
        # input.(batch, problem, embedding)
        return self.W2(F.relu(self.W1(input1))), 0


 