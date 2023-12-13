import torch
import torch.nn as nn
from recbole.utils import set_color
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.model.init import xavier_normal_initialization
    
class PathEncoder(nn.Module):
    def __init__(self, kg_embedding_size, hidden_size):
        super(PathEncoder, self).__init__()
        self.bilstm = nn.LSTM(kg_embedding_size, hidden_size, batch_first=True, bidirectional=True)
        
    def forward(self, path_emb_seq, valid_seq_len):
        path_vecs, _ = self.bilstm(path_emb_seq)
        f_path_vecs, b_path_vecs = torch.split(path_vecs, path_vecs.shape[-1] // 2, dim=-1)
        # mean pooling for all path
        u_path_vecs = torch.mean(f_path_vecs, dim=1)
        i_path_vecs = torch.mean(b_path_vecs, dim=1)

        return u_path_vecs, i_path_vecs

class KAERR(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KAERR, self).__init__(config, dataset)
        self.USER_SENTS = config['USER_DOC_FIELD']
        self.ITEM_SENTS = config['ITEM_DOC_FIELD']
        self.neg_prefix = config['NEG_PREFIX']
        self.USER_ID_FIELD = config['USER_ID_FIELD']
        self.ITEM_ID_FIELD = config['ITEM_ID_FIELD']
        self.USE_PATH = config['USE_PATH']
        self.CASE_STUDY = config['CASE_STUDY']
        self.USE_SINGLE_INTER = config['USE_SINGLE_INTER']
        self.fusion_embedding_size = config['fusion_embedding_size']

        if self.USE_SINGLE_INTER:
            self.user_single_inter = torch.from_numpy(
                dataset.user_single_inter_matrix(form='coo').todense()).bool().to(self.device)

        self.user_entity_type = config['USER_ENTITY_TYPE']
        self.item_entity_type = config['ITEM_ENTITY_TYPE']
        self.ntype_start_idx = dataset.ntype_start_idx
        self.kg_embedding_size = config['kg_embedding_size']
        self.hidden_size = config['hidden_size']
        if config['USE_KGE']:
            self.kg_embedding_num = dataset.kg_embedding.shape[0]
        else:
            self.kg_embedding_num = dataset.kg.num_nodes() + len(dataset.kg.etypes) + 1
        self.kg_emb = nn.Embedding(self.kg_embedding_num, self.kg_embedding_size, padding_idx=0)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(self.kg_embedding_size)

        #self.kg_embedding_size = dataset.kg_embedding.shape[1]
        self.max_path_num = config['max_path_num']

        # path encoding
        self.path_encoder = PathEncoder(self.kg_embedding_size, self.hidden_size)

        self.user_path_attn = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        self.item_path_attn = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        self.user_path_predict = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        self.item_path_predict = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()
        self.bpr_loss = BPRLoss()
        self.apply(xavier_normal_initialization)
        if config['USE_KGE']:
            self.kg_emb.weight.data.copy_(torch.from_numpy(dataset.kg_embedding))
        

    def forward(self, user_ids, item_ids, user_sents, item_sents, path, valid_path_length, valid_path_num):
        # KG path encoder
        path_emb_seq = self.kg_emb(path).view(-1, path.shape[-1], self.kg_embedding_size)  # bs*max_path_num, max_path_len, kg_emb_size                 
        u_path_emb, i_path_emb = self.path_encoder(path_emb_seq, valid_path_length.view(-1))    # bs*max_path_num, kg_emb_size
        # self-attention between paths
        valid_path_mask = torch.arange(path.shape[1]).unsqueeze(0).repeat(path.shape[0], 1).to(self.device) >= valid_path_num.unsqueeze(1)
        u_path_emb = u_path_emb.reshape(-1, self.hidden_size)
        i_path_emb = i_path_emb.reshape(-1, self.hidden_size)
        u_path_weight = self.user_path_attn(u_path_emb)
        i_path_weight = self.item_path_attn(i_path_emb)
        u_path_weight = u_path_weight.view(path.shape[0], -1)
        i_path_weight = i_path_weight.view(path.shape[0], -1)
        # sum pooling
        u_path_A = torch.where(valid_path_mask == 1, 0, u_path_weight)
        i_path_A = torch.where(valid_path_mask == 1, 0, i_path_weight)
        u_path_emb_pool = torch.matmul(u_path_A.unsqueeze(1), u_path_emb.reshape(-1, path.shape[1], self.hidden_size)).squeeze(1)
        i_path_emb_pool = torch.matmul(i_path_A.unsqueeze(1), i_path_emb.reshape(-1, path.shape[1], self.hidden_size)).squeeze(1)
        u_path_score_final = self.user_path_predict(u_path_emb_pool)
        i_path_score_final = self.item_path_predict(i_path_emb_pool)
        score = u_path_score_final + i_path_score_final
        score = score / 2
        
        return score.squeeze(), (u_path_score_final, [0], u_path_A, i_path_score_final, [0], i_path_A)

    def calculate_loss(self, interaction):
        user_ids = interaction[self.USER_ID]
        neg_user_ids = interaction[self.neg_prefix + self.USER_ID]
        item_ids = interaction[self.ITEM_ID]
        neg_item_ids = interaction[self.neg_prefix + self.ITEM_ID]

        user_sents = interaction[self.USER_SENTS]
        neg_user_sents = interaction[self.neg_prefix + self.USER_SENTS]
        item_sents = interaction[self.ITEM_SENTS]
        neg_item_sents = interaction[self.neg_prefix + self.ITEM_SENTS]

        
        if self.USE_PATH:
            user_item_path = interaction['user_item_paths']
            user_item_valid_path_length = interaction['user_item_valid_path_length']
            user_item_valid_path_num = interaction['user_item_valid_path_num']
            user_neg_item_path = interaction['user_neg_item_paths']
            user_neg_item_valid_path_length = interaction['user_neg_item_valid_path_length']
            user_neg_item_valid_path_num = interaction['user_neg_item_valid_path_num']
            neg_user_item_path = interaction['neg_user_item_paths']
            neg_user_item_valid_path_length = interaction['neg_user_item_valid_path_length']
            neg_user_item_valid_path_num = interaction['neg_user_item_valid_path_num']
            pos_score, pos_record = self.forward(user_ids-1, item_ids-1, user_sents, item_sents, user_item_path, user_item_valid_path_length, user_item_valid_path_num)
            neg1_score, neg1_record = self.forward(user_ids-1, neg_item_ids-1, user_sents, neg_item_sents, user_neg_item_path, user_neg_item_valid_path_length, user_neg_item_valid_path_num)
            neg2_score, neg2_record = self.forward(neg_user_ids-1, item_ids-1, neg_user_sents, item_sents, neg_user_item_path, neg_user_item_valid_path_length, neg_user_item_valid_path_num)
        
        else:
            pos_score, _ = self.forward(user_ids-1, item_ids-1, user_sents, item_sents, None, None, None)
            neg1_score, _ = self.forward(user_ids-1, neg_item_ids-1, user_sents, neg_item_sents, None, None, None)
            neg2_score, _ = self.forward(neg_user_ids-1, item_ids-1, neg_user_sents, item_sents, None, None, None)

        if self.USE_SINGLE_INTER:
            neg1_user_single_inter = self.user_single_inter[user_ids, neg_item_ids]
            neg2_user_single_inter = self.user_single_inter[neg_user_ids, item_ids]

            neg1_user_score, neg1_item_score = neg1_record[0], neg1_record[3]
            neg2_user_score, neg2_item_score = neg2_record[0], neg2_record[3]

            single_loss_1 = self.bpr_loss(neg1_user_score*neg1_user_single_inter, neg1_item_score*neg1_user_single_inter)
            single_loss_2 = self.bpr_loss(neg2_user_score*neg2_user_single_inter, neg2_item_score*neg2_user_single_inter)

            loss = self.bpr_loss(2*pos_score, neg1_score+neg2_score) + 0.5*single_loss_1 + 0.5*single_loss_2

        else:
            loss = self.bpr_loss(2*pos_score, neg1_score+neg2_score)

        return loss

    def predict(self, interaction):
        user_ids = interaction[self.USER_ID] - 1
        item_ids = interaction[self.ITEM_ID] - 1
        user_sents = interaction[self.USER_SENTS]
        item_sents = interaction[self.ITEM_SENTS]

        if self.USE_PATH:
            user_item_path = interaction['user_item_paths']
            user_item_valid_path_length = interaction['user_item_valid_path_length']
            user_item_valid_path_num = interaction['user_item_valid_path_num']
            score, record = self.forward(user_ids, item_ids, user_sents, item_sents, user_item_path, user_item_valid_path_length, user_item_valid_path_num)
        else:
            score, _ = self.forward(user_ids, item_ids, user_sents, item_sents, None, None, None)

        return score