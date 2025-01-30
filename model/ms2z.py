
import torch
import torch.nn as nn

from .graph_lib import *
from .chem_encoder import *
from .decoder import HierGATDecoder
from .transformer import *

class Ms2z(nn.Module):
    def __init__(
            self, vocab_data, max_seq_len, node_dim, edge_dim, atom_layer_lstm_iterations,
            chem_encoder_h_size,
            latent_dim,
            memory_seq_len,
            decoder_layers, decoder_heads, decoder_ff_dim, decoder_dropout,
            target_var=1.0,
            tgt_token_priority_weight=0.2,
            ):
        assert (node_dim+edge_dim) % decoder_heads == 0, "node_dim+edge_dim must be divisible by decoder_heads."
        super(Ms2z, self).__init__()
        attached_motif_index_map = vocab_data['attached_motif_index_map']
        bonding_cnt_tensor = vocab_data['bonding_cnt_tensor']
        atom_layer_list = vocab_data['atom_layer_list']
        bos_idx = vocab_data['bos']
        pad_idx = vocab_data['pad']
        unk_idx = vocab_data['unk']

        self.is_sequential = False # Sequential processing
        self.use_chem_encoder = True


        self.max_seq_len = max_seq_len
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim

        self.tgt_token_priority_weight = tgt_token_priority_weight

        # Vocabulary embedding
        self.frag_embedding = FragEmbeddings(
            node_dim, edge_dim,
            attached_motif_index_map, 
            bonding_cnt_tensor, 
            atom_layer_list,
            atom_layer_lstm_iterations,
            bos_idx, pad_idx, unk_idx,
            )
        
        self.motif_size = self.frag_embedding.motif_size
        self.att_size = self.frag_embedding.att_size
        self.max_bonding_cnt = self.frag_embedding.max_bonding_cnt

        self.bos = nn.Parameter(torch.tensor(bos_idx, dtype=torch.int32), requires_grad=False)
        self.attached_bos_idx = nn.Parameter(torch.tensor([self.bos, 0], dtype=torch.int64), requires_grad=False)
        self.pad = nn.Parameter(torch.tensor(pad_idx, dtype=torch.int32), requires_grad=False)
        self.attached_pad_idx = nn.Parameter(torch.tensor([self.pad, 0], dtype=torch.int64), requires_grad=False)
        self.unk = nn.Parameter(torch.tensor(unk_idx, dtype=torch.int32), requires_grad=False)
        self.attached_unk_idx = nn.Parameter(torch.tensor([self.unk, 0], dtype=torch.int64), requires_grad=False)

        # Structure encoder
        self.chem_encoder = StructureEncoder(node_dim=node_dim+edge_dim, edge_dim=edge_dim, h_size=chem_encoder_h_size)

        # MS encoder
        self.ms_encoder = None

        # Latent sampler
        self.chem_latent_sampler = LatentSampler(input_dim=chem_encoder_h_size, latent_dim=latent_dim)
        self.memory_linear = nn.Linear(latent_dim, (node_dim+edge_dim)*memory_seq_len)
        self.target_var = target_var
        self.memory_seq_len = memory_seq_len

        # Transformer decoder
        self.decoder = HierGATDecoder(
            num_layers=decoder_layers,
            node_dim=node_dim+edge_dim,
            edge_dim=edge_dim,
            num_heads=decoder_heads,
            ff_dim=decoder_ff_dim,
            dropout=decoder_dropout,
        )
        
        self.predict_linear = nn.Linear(node_dim+edge_dim, node_dim+edge_dim)
        
        # self.pred_token_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='none')
        
        # predict motif
        self.pred_motif_layer = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, self.motif_size),
        )
        self.pred_motif_loss_fn = F.cross_entropy

        # predict attachment
        self.pred_attachment_layer = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, self.att_size),
        )
        self.pred_attachment_loss_fn = F.cross_entropy

        # predict root bond position
        self.pred_root_bond_pos_loss_fn = F.cross_entropy
        self.pred_root_bond_pos_layer = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, self.max_bonding_cnt),
        )

        # # Vocab embedding loss
        # self.vocab_embed_linear = nn.Linear(node_dim, node_dim)

        # # self.fp_linear1 = nn.Linear(node_dim+edge_dim, 512)
        # # self.fp_linear2 = nn.Linear(512, 167)

    def forward(self, input_tensor, target_tensor):
        """
        Forward pass for Ms2z model.

        Args:
            token_tensor (torch.Tensor): Token input tensor. [batch_size, seq_len, 2 (motif, attachment)]
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. [batch_size, seq_len, 3 (parent_idx, parent_bond_pos, bond_pos)]
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.
        """
        assert 'token' in target_tensor and 'order' in target_tensor and 'mask' in target_tensor, "target tensor must contain 'token', 'order', and 'mask' keys."

        self.frag_embedding.reset_calc_embeddings()

        # Prepare input
        if self.use_chem_encoder:
            node_embed_with_root_bond, parent_bond_pos_embed = self.prepare_chem_input(*[input_tensor[key] for key in ['token', 'order', 'mask']])
            encoder_output = self.chem_encoder(
                node_embed=node_embed_with_root_bond, 
                edge_attr=parent_bond_pos_embed, 
                order_tensor=input_tensor['order'], 
                mask_tensor=input_tensor['mask'],
            )
            z, mean, log_var = self.chem_latent_sampler(encoder_output)
            
        else:
            pass

        # Prepare decoder
        if self.is_sequential:
            pass
        else:
            node_embed_flat, edge_index_flat, edge_attr_embed_flat, \
                tgt_idx_flat, mask_seq_flat, mask_edge_flat, batch_idx_flat, unk_idx_flat = \
                self.prepare_batch_decoder(*[target_tensor[key] for key in ['token', 'order', 'mask']])
            
        # Transformer decoder
        memory = z # [batch_size, latent_dim]
        memory = self.memory_linear(memory) # [batch_size, (node_dim+edge_dim)*memory_seq_len]
        memory_flat = memory[batch_idx_flat] # [valid_total, (node_dim+edge_dim)*memory_seq_len]
        memory_flat = memory_flat.reshape(*memory_flat.shape[:-1], -1, self.memory_seq_len) # [valid_total, node_dim+edge_dim, memory_seq_len]
        memory_flat = memory_flat.transpose(1, 2) # [valid_total, memory_seq_len, node_dim+edge_dim]

        # Decode using Transformer Decoder
        memory_mask_flat = torch.zeros(*memory_flat.shape[:2], dtype=torch.bool, device=memory_flat.device) # [valid_total, memory_seq_len, node_dim+edge_dim]

        # Decoder
        decoder_output = self.decoder(
            x=node_embed_flat, 
            edge_index=edge_index_flat, 
            edge_attr=edge_attr_embed_flat, 
            enc_output=memory_flat, 
            tgt_mask=mask_seq_flat, 
            memory_mask=memory_mask_flat, 
            edge_mask=mask_edge_flat
        ) # Shape: [valid_total, seq_len, node_dim+edge_dim]

        output_flat = decoder_output[torch.arange(decoder_output.size(0)), unk_idx_flat] # Shape: [valid_total, node_dim+edge_dim]
        output_flat = self.predict_linear(output_flat) # Shape: [valid_total, node_dim+edge_dim]

        node_embed_flat = output_flat[:, :self.node_dim] # Shape: [valid_total, node_dim]
        edge_attr_flat = output_flat[:, self.node_dim:] # Shape: [valid_total, edge_dim]

        # Predict motif
        tgt_motif_idx_flat = tgt_idx_flat[:, 0] # Shape: [valid_total]
        pred_motif_loss, pred_motif_acc = self.calc_predict_motif_loss(node_embed_flat, tgt_motif_idx_flat)

        # KL divergence loss
        kl_divergence_loss = self.chem_latent_sampler.calc_kl_divergence(mean, log_var, target_var=self.target_var)

        loss_list = {
            'KL': kl_divergence_loss,
            'pred_motif': pred_motif_loss,
        }
        acc_list = {
            'pred_motif': pred_motif_acc.item(),
        }
        target_data = {
            'KL': {'loss': kl_divergence_loss.item(), 'accuracy': None, 'criterion': 'KL Divergence'},
            'pred_motif': {'loss': pred_motif_loss.item(), 'accuracy': pred_motif_acc.item(), 'criterion': get_criterion_name(self.pred_motif_loss_fn)},
        }

        return loss_list, acc_list, target_data
    
    def enable_sequential(self):
        self.is_sequential = True
    
    def disable_sequential(self):
        self.is_sequential = False

    def enable_chem_encoder(self):
        self.use_chem_encoder = True

    def enable_ms_encoder(self):
        assert self.ms_encoder is not None, "MS encoder is not defined."
        self.use_chem_encoder = False

    def get_parent_attached_motif_idx(self, token_tensor, order_tensor):
        mask = order_tensor[..., 0] < 0
        valid_parent_idx = order_tensor[..., 0].masked_fill(mask, 0)
        parent_attached_motif_idx = torch.gather(
            token_tensor, dim=-2,
            index=valid_parent_idx.unsqueeze(-1).expand(*token_tensor.shape[:-1], 2)
        ) # [batch_size, seq_len, 2]
        parent_attached_motif_idx[mask] = self.attached_pad_idx.expand_as(parent_attached_motif_idx[mask])
        return parent_attached_motif_idx

    def prepare_chem_input(self, token_tensor, order_tensor, mask_tensor):
        attached_motif_idx = token_tensor[...,:] # [batch_size, seq_len, 2]
        root_bond_pos = order_tensor[...,2] # [batch_size, seq_len]
        attached_motif_idx_with_bond_pos = torch.cat([attached_motif_idx, root_bond_pos.unsqueeze(-1)], dim=-1) # [batch_size, seq_len, 3]
        node_embed_with_root_bond = self.frag_embedding(attached_motif_idx_with_bond_pos) # [batch_size, seq_len, node_dim+edge_dim]

        parent_attached_motif_idx = self.get_parent_attached_motif_idx(token_tensor, order_tensor)
        parent_bond_pos = order_tensor[...,1] # [batch_size, seq_len]
        parent_attached_motif_idx_with_bond_pos = torch.cat([parent_attached_motif_idx, parent_bond_pos.unsqueeze(-1)], dim=-1) # [batch_size, seq_len, 3]
        parent_bond_pos_embed = self.frag_embedding.embed_edge_attr(parent_attached_motif_idx_with_bond_pos) # [batch_size, seq_len, max_bonding_cnt]

        return node_embed_with_root_bond, parent_bond_pos_embed

    def prepare_batch_decoder(self, token_tensor, order_tensor, mask_tensor):
        node_idx_flat, edge_index_flat, edge_attr_idx_flat, \
            tgt_idx_flat, mask_seq_flat, mask_edge_flat, batch_idx_flat, unk_idx_flat = \
            self.get_decoder_input_idx(token_tensor, order_tensor, mask_tensor)
        
        node_embed_flat = self.frag_embedding(node_idx_flat) # [valid_total, seq_len, node_dim+edge_dim]
        edge_attr_embed_flat = self.frag_embedding.embed_edge_attr(edge_attr_idx_flat) # [valid_total, seq_len, edge_dim]

        return node_embed_flat, edge_index_flat, edge_attr_embed_flat, tgt_idx_flat, mask_seq_flat, mask_edge_flat, batch_idx_flat, unk_idx_flat


    def get_decoder_input_idx(self, token_tensor, order_tensor, mask_tensor):
        batch_size, seq_len, _ = token_tensor.shape

         # **1. Create padding and unknown order tensors**
        pad_order = torch.full((order_tensor.size(2),), -1, dtype=order_tensor.dtype, device=order_tensor.device)
        unk_order = torch.full((order_tensor.size(2),), -1, dtype=order_tensor.dtype, device=order_tensor.device)
        false_mask = torch.tensor(False, dtype=mask_tensor.dtype, device=mask_tensor.device)
        # true_mask = torch.tensor(True, dtype=mask_tensor.dtype, device=mask_tensor.device)
        
        # **2. Add <BOS> token at the beginning of each sequence**
        token_seq_tensor = torch.cat([
            self.attached_bos_idx.repeat(batch_size, 1).unsqueeze(1),
            token_tensor
        ], dim=1)  # Shape: [batchsize, seq_len+1, 2]
        order_seq_tensor = torch.cat([
            pad_order.unsqueeze(0).repeat(batch_size, 1, 1),
            order_tensor
        ], dim=1)  # Shape: [batchsize, seq_len+1, 3]
        mask_seq_tensor = torch.cat([
            false_mask.unsqueeze(0).repeat(batch_size, 1),
            mask_tensor,
        ], dim=1)  # Shape: [batchsize, seq_len+1]

        # **3. Apply <PAD> token where mask is True**
        token_seq_tensor[mask_seq_tensor] = self.attached_pad_idx # Shape: [batchsize, seq_len+1, 2]
        order_seq_tensor[mask_seq_tensor] = pad_order # Shape: [batchsize, seq_len+1, 3]
        
        # **4. Create valid input mask for sequence expansion**
        valid_input_idx = ~Ms2z.create_expand_mask(mask_tensor, fill=True) # Shape: [batch_size, seq_len, seq_len]
        valid_input_idx = torch.cat([valid_input_idx, false_mask.expand(*valid_input_idx.shape[:2], 1)], dim=-1)  # Shape: [batch_size, seq_len, seq_len+1]
        
        # Expand valid indices for different dimensions
        valid_input_idx2 = valid_input_idx.unsqueeze(3).expand(-1, -1, -1, 2)  # Shape: [batch_size, seq_len, seq_len, 2]
        valid_input_idx3 = valid_input_idx.unsqueeze(3).expand(-1, -1, -1, 3)  # Shape: [batch_size, seq_len, seq_len, 3]

        # **5. Create unknown token mask for sequence expansion**
        valid_diagonal_idx = ~Ms2z.create_expand_mask(mask_tensor, fill=False)  # Shape: [batch_size, seq_len, seq_len]
        valid_unk_idx = torch.cat([false_mask.unsqueeze(0).unsqueeze(0).expand(batch_size,seq_len,1), valid_diagonal_idx], dim=-1)  # Shape: [batch_size, seq_len, seq_len+1]
        valid_unk_idx2 = valid_unk_idx.unsqueeze(3).expand(-1, -1, -1, 2)  # Shape: [batch_size, seq_len, seq_len+1, 2]
        valid_unk_idx3 = valid_unk_idx.unsqueeze(3).expand(-1, -1, -1, 3)  # Shape: [batch_size, seq_len, seq_len+1, 3]
        
        # Get valid sequence indices
        valid_indices = (~mask_tensor).nonzero(as_tuple=True)  # (2 [batch, seq], valid_idx_total)

        unk_idx_flat = torch.nonzero(valid_unk_idx[valid_indices[0], valid_indices[1]], as_tuple=True)[1]

        # **6. Compute mask and valid sequence lengths**
        mask_seq_flat = valid_input_idx[valid_indices[0], valid_indices[1]] # Shape: [valid_idx_total, seq_len+1]
        mask_seq_flat[:,-1] = True
        mask_seq_flat = ~mask_seq_flat.roll(1) # Shape: [valid_idx_total, seq_len+1]
        valid_length_flat = (~mask_seq_flat).sum(dim=-1) # Shape: [valid_idx_total]

        # **7. Expand token sequence with valid indices**
        token_seq_ex = self.attached_pad_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, seq_len+1, 1)  # Shape: [batch_size, seq_len, seq_len+1, 2]
        token_seq_ex = torch.where(valid_input_idx2, token_seq_tensor.unsqueeze(1), token_seq_ex)  # Shape: [batch_size, seq_len, seq_len+1, 2]
        token_seq_ex = torch.where(valid_unk_idx2, self.attached_unk_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, seq_len+1, 1), token_seq_ex)  # Shape: [batch_size, seq_len, seq_len+1, 2]
        token_seq_flat = token_seq_ex[valid_indices[0], valid_indices[1]] # Shape: [valid_idx_total, seq_len+1, 2]
        
        # **8. Expand order sequence with valid indices**
        order_seq_ex = pad_order.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, seq_len+1, 1)  # Shape: [batch_size, seq_len, seq_len+1, 3]
        order_seq_ex = torch.where(valid_input_idx3, order_seq_tensor.unsqueeze(1), order_seq_ex)  # Shape: [batch_size, seq_len, seq_len+1, 3]
        order_seq_ex = torch.where(valid_unk_idx3, unk_order.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, seq_len+1, 1), order_seq_ex)  # Shape: [batch_size, seq_len, seq_len+1, 3]
        order_seq_flat = order_seq_ex[valid_indices[0], valid_indices[1]+1] # Shape: [valid_idx_total, seq_len+1, 1]

        # cat node_idx and root_bond_pos
        node_idx_flat = torch.cat([token_seq_flat, order_seq_flat[...,[2]]], dim=-1) # Shape: [valid_idx_total, seq_len+1, 3]

        # **10. Construct edge indices for parent-child relationships**
        parent_idx_flat = order_seq_flat[..., [0]]+1 # Shape: [valid_idx_total, seq_len+1, 1]
        parent_token_idx_flat = torch.gather(token_seq_flat, dim=1, index=parent_idx_flat.expand(-1, -1, 2)) # Shape: [valid_idx_total, seq_len+1, 2]
        parent_idx_flat[:,0] = -1
        parent_idx_flat[mask_seq_flat] = -1
        parent_token_idx_flat[:,0] = self.attached_pad_idx
        parent_token_idx_flat[mask_seq_flat] = self.attached_pad_idx

        # **11. Construct edge index tensor**
        mask_edge_flat = mask_seq_flat[:,1:] # Shape: [valid_idx_total, seq_len]
        # `src` → Parent nodes
        src = parent_idx_flat[:,1:]  # Shape: [valid_total, seq_len, 1]
        # `dst` → Child nodes (current token index)
        dst = torch.arange(1, seq_len+1, device=order_tensor.device, dtype=order_tensor.dtype).unsqueeze(0).repeat(src.size(0), 1)  # Shape: [valid_total, seq_len]
        dst[mask_edge_flat] = -1
        dst = dst.unsqueeze(-1)  # Shape: [valid_total, seq_len, 1]
        edge_index_flat = torch.cat([src, dst], dim=-1)  # Shape: [valid_total, seq_len, 2]

        # **12. Construct edge attribute index tensor**
        # `edge_attr_idx_flat` → Parent nodes and bond position
        edge_attr_idx_flat = torch.cat([parent_token_idx_flat, order_seq_flat[...,[2]]], dim=-1) # Shape: [valid_idx_total, seq_len+1, 3]
        edge_attr_idx_flat = edge_attr_idx_flat[:,1:] # Shape: [valid_idx_total, seq_len, 3]

        # **13. Target indices for decoder prediction**
        tgt_attached_motif_idx_flat = token_seq_tensor[valid_indices[0], valid_length_flat-1] # Shape: [valid_idx_total, 2]
        tgt_root_bond_pos_flat = order_seq_tensor[valid_indices[0], valid_length_flat-1, 2].unsqueeze(-1) # Shape: [valid_idx_total,1]
        tgt_idx_flat = torch.cat([tgt_attached_motif_idx_flat, tgt_root_bond_pos_flat], dim=-1) # Shape: [valid_idx_total, 3]

        # **14. Get batch indices**
        batch_idx_flat = valid_indices[0] # Shape: [valid_idx_total]


        node_idx_flat = node_idx_flat[:, :-1, :] # Shape: [valid_idx_total, seq_len, 3]
        edge_index_flat = edge_index_flat[:, :-1, :] # Shape: [valid_idx_total, seq_len-1, 2]
        edge_attr_idx_flat = edge_attr_idx_flat[:, :-1, :] # Shape: [valid_idx_total, seq_len-1, 3]
        mask_seq_flat = mask_seq_flat[:, :-1] # Shape: [valid_idx_total, seq_len]
        mask_edge_flat = mask_edge_flat[:, :-1] # Shape: [valid_idx_total, seq_len-1]
        
        return node_idx_flat, edge_index_flat, edge_attr_idx_flat, tgt_idx_flat, mask_seq_flat, mask_edge_flat, batch_idx_flat, unk_idx_flat
    
    def prepare_sequentail_input(self):
        pass
    
    @staticmethod
    def create_expand_mask(mask, fill=True):
        """
        Expand a mask from [batch_size, seq_len] to [batch_size, seq_len, seq_len]
        and set positions where j > i (future positions) to False.

        Args:
            mask (torch.Tensor): Input mask of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Expanded mask of shape [batch_size, seq_len, seq_len].
        """
        batch_size, seq_len = mask.size()
        
        if fill:
            # Create future mask: [seq_len, seq_len], True where j <= i
            seq_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=mask.device))
        else:
            # Create future mask: [seq_len, seq_len], True where j == i
            seq_mask = torch.eye(seq_len, dtype=torch.bool, device=mask.device)
        
        # Expand input mask to [batch_size, seq_len, seq_len]
        expand_mask = mask.unsqueeze(-1).expand(-1, -1, seq_len)
        
        # Combine with the future mask
        expand_mask = ~expand_mask & seq_mask
        return ~expand_mask
    
    def calc_fp_loss(self, z, fp_tensor):
        # fingerprint loss
        fp_x = self.z_fp_layer(z)
        # fp_x = self.fp_dropout(fp_x)
        fp_x_binary = torch.where(fp_x < 0.5, torch.tensor(0.0, device=fp_x.device), torch.tensor(1.0, device=fp_x.device))

        # fp_x = self.vocab_embedding(target_tensor[:, 0])
        # fp_x = self.fp_linear1(fp_x)
        # fp_x = torch.relu(fp_x)
        # fp_x = self.fp_linear2(fp_x)

        element_wise_loss  = self.z_fp_loss_fn(fp_x, fp_tensor)
        fp_acc = self.z_fp_loss_fn(fp_x_binary, fp_tensor).mean()

        # Create masks for 0 and 1 in fp_tensor
        zero_mask = (fp_tensor == 0)  # Shape: [batch_size, fp_size]
        one_mask = (fp_tensor == 1)  # Shape: [batch_size, fp_size]

        # Compute the mean loss for zero_mask and one_mask
        zero_loss_mean = torch.sum(element_wise_loss * zero_mask, dim=1) / (zero_mask.sum(dim=1) + 1e-8)  # Shape: [batch_size]
        one_loss_mean = torch.sum(element_wise_loss * one_mask, dim=1) / (one_mask.sum(dim=1) + 1e-8)    # Shape: [batch_size]

        # Compute the final loss as the average of zero_loss_mean and one_loss_mean
        fp_loss = (zero_loss_mean + one_loss_mean).mean()
        # fp_loss = element_wise_loss.mean()

        return fp_loss

    def calc_predict_motif_loss(self, node_embed, tgt_motif_id):
        # Project node embeddings to vocabulary logits
        logits_vocab = self.pred_motif_layer(node_embed)  # Shape: [data_size, node_dim] -> [data_size, vocab_size]
        # Apply Softmax to get probabilities
        # probabilities = F.softmax(logits_vocab, dim=-1)  # Shape: [data_size, vocab_size]
        
        losses = self.pred_motif_loss_fn(logits_vocab, tgt_motif_id, reduction='none')

        # Compute weights for each token
        unique_tokens, counts = torch.unique(tgt_motif_id, return_counts=True)
        token_weights = (1 / counts.float()) / unique_tokens.size(0)

        # Create a weight tensor for each target
        weight_map = {token.item(): weight for token, weight in zip(unique_tokens, token_weights)}
        weights = torch.tensor([weight_map[token.item()] for token in tgt_motif_id], device=node_embed.device)  # Shape: [data_size]

        # Apply weights to the individual losses
        weighted_losses = losses * weights  # Shape: [data_size]

        # Compute the weighted average loss
        loss = weighted_losses.sum()

        # Compute the accuracy
        predicted_token_id = torch.argmax(logits_vocab, dim=1)
        correct = (predicted_token_id == tgt_motif_id).float()
        accuracy = correct.mean()
        # tgt_token_id[torch.where(correct >= 1.0)[0]]

        return loss, accuracy

    def calc_predict_attachment_loss(self, node_embed, tgt_token_id):
        pass
    
    def calc_precdict_parent_joint_loss(self, joint_edge_embed, tgt_parent_joint, tgt_bond_type):
        if joint_edge_embed.size(0) == 0:
            return torch.tensor(0.0, device=joint_edge_embed.device), torch.tensor(0.0, device=joint_edge_embed.device)
        # Project node embeddings to vocabulary logits
        joint_edge_embed = self.parent_joint_linear(joint_edge_embed)

        logits_parent_joint_pos = self.parent_joint_pos_layer(joint_edge_embed)  # Shape: [data_size, joint_edge_dim] -> [data_size, poteintial_joint_size]
        pos_loss = self.pred_parent_joint_loss_fn(logits_parent_joint_pos, tgt_parent_joint)

        predicted_parent_joint_pos = torch.argmax(logits_parent_joint_pos, dim=1)
        pjp_correct = (predicted_parent_joint_pos == tgt_parent_joint).float()
        pjp_accuracy = pjp_correct.mean()

        logits_parent_joint_bond_type = self.parent_bond_type_layer(joint_edge_embed)  # Shape: [data_size, joint_edge_dim] -> [data_size, 3]
        bond_type_loss = self.pred_parent_joint_loss_fn(logits_parent_joint_bond_type, tgt_bond_type)
        
        predicted_parent_joint_bond_type = torch.argmax(logits_parent_joint_bond_type, dim=1)
        pjb_correct = (predicted_parent_joint_bond_type == tgt_bond_type).float()
        pjb_accuracy = pjb_correct.mean()

        loss = pos_loss + bond_type_loss
        accuracy = (pjp_correct * pjb_correct).mean()

        return loss, accuracy

    def calc_node_embed_loss(self, node_embed, tgt_token_id_flat):
        node_embed = self.vocab_embed_linear(node_embed)
        return self.frag_embedding.calc_node_loss(node_embed, tgt_token_id_flat)

    def get_detail_loss(self, vocab_tensor, order_tensor, mask_tensor, fp_tensor):
        """
        Forward pass for Ms2z model.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. [batch_size, seq_len, 5 (parent_idx, parent_atom_pos, atom_pos, bond_type[1~3], level)]
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.
        """
        node_embed, edge_attr, token_seq_tensor, edge_index, mask_tensor_ex, \
            target_tensor, parent_token_seq_tensor, root_atom_pos_tensor, \
                tgt_root_atom_pos_tensor, joint_atom_pos_tensor, tgt_joint_atom_pos_tensor, \
                    root_bond_type_tensor, tgt_root_bond_type_tensor, joint_bond_type_tensor, \
                        unk_nodes = self.prepare_input(vocab_tensor, order_tensor, mask_tensor)

        # Calculate latent variable
        z, mean, log_var = self.calc_chem_z(node_embed[:,1:], edge_attr[:,1:], order_tensor, mask_tensor)

        # Transformer decoder
        memory = z.unsqueeze(1) # [batch_size, 1, latent_dim]
        memory = self.memory_linear(memory) # [batch_size, 1, node_dim+edge_dim]

        # Decode using Transformer Decoder
        memory_mask = torch.zeros(memory.size(0), memory.size(1), dtype=torch.bool).to(memory.device)
        memory_mask[:, 0] = True

        # node_dim + edge_dim + joint_edge_dim
        # [batch_size, seq_len+1, node_dim+edge_dim] -> [batch_size, seq_len+1, node_dim+edge_dim+joint_edge_dim]
        node_embed_with_joint = torch.cat([node_embed, torch.zeros(node_embed.shape[:-1]+(self.joint_edge_dim,), dtype=node_embed.dtype, device=node_embed.device)], dim=-1)
        node_joint_potentials = self.frag_embedding.joint_potential[parent_token_seq_tensor][...,1]

        # Decoder
        decoder_output = self.decoder(
            x=node_embed_with_joint,
            edge_index=edge_index,
            edge_attr=node_joint_potentials,
            enc_output=memory,
            unk_feature=unk_nodes,
            tgt_mask=mask_tensor_ex,
            memory_mask=memory_mask
        ) # Shape: [batch_size, seq_len, node_dim+edge_dim+joint_edge_dim]

        # Flatten for output layer
        output_flat = decoder_output.view(-1, decoder_output.size(-1)) # Shape: [batch_size*seq_len, node_dim+edge_dim+joint_edge_dim]
        
        tgt_token_id_flat = target_tensor.view(-1)
        # tgt_bond_pos_tensor = torch.cat([order_tensor[:,:,2], padding], dim=1)
        # tgt_bond_pos_flat = tgt_bond_pos_tensor.view(-1)
        valid_mask = tgt_token_id_flat != self.pad
        valid_mask_without_eos = valid_mask & (tgt_token_id_flat != self.eos) & (tgt_root_atom_pos_tensor.view(-1)!=-1)


        tgt_root_atom_pos_flat = tgt_root_atom_pos_tensor.view(-1)
        tgt_parent_atom_pos_flat = tgt_joint_atom_pos_tensor.view(-1)
        tgt_bond_type_flat = tgt_root_bond_type_tensor.view(-1)

        output_flat_without_eos = output_flat[valid_mask_without_eos]
        output_flat = output_flat[valid_mask]
        tgt_token_id_flat = tgt_token_id_flat[valid_mask]
        tgt_root_atom_pos_flat = tgt_root_atom_pos_flat[valid_mask_without_eos]
        tgt_parent_atom_pos_flat = tgt_parent_atom_pos_flat[valid_mask_without_eos]
        tgt_bond_type_flat = tgt_bond_type_flat[valid_mask_without_eos]
        
        node_embed_flat = output_flat[:,:self.node_dim] # [batch_size*seq_len, node_dim]
        root_edge_embed_flat = output_flat_without_eos[:,self.node_dim:self.node_dim+self.edge_dim] # [batch_size*seq_len, edge_dim]
        parent_joint_edge_embed_flat = output_flat_without_eos[:,self.node_dim+self.edge_dim:] # [batch_size*seq_len, joint_edge_dim]
        
        # calc token loss
        predict_token_loss, predict_token_acc = self.calc_predict_token_loss(node_embed_flat, tgt_token_id_flat, detail=True)


        return tgt_token_id_flat, predict_token_loss, predict_token_acc


    def get_config_param(self):
        """
        Get model configuration.

        Returns:
            dict: Model configuration.
        """
        return {
            'max_seq_len': self.max_seq_len,
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'joint_edge_dim': self.joint_edge_dim,
            'latent_dim': self.latent_dim,
            'decoder_layers': self.decoder_layers,
            'decoder_heads': self.decoder_heads,
            'decoder_ff_dim': self.decoder_ff_dim,
            'decoder_dropout': self.decoder_dropout,
            'fp_dim': self.fp_dim,
            'target_var': self.target_var,
        }
    
    @staticmethod
    def from_config_param(config_param):
        """
        Create model from configuration parameters.

        Args:
            config_param (dict): Model configuration parameters.

        Returns:
            Ms2z: Model instance.
        """
        return Ms2z(
            vocab_data=config_param['vocab_data'],
            max_seq_len=config_param['max_seq_len'],
            node_dim=config_param['node_dim'],
            edge_dim=config_param['edge_dim'],
            joint_edge_dim=config_param['joint_edge_dim'],
            latent_dim=config_param['latent_dim'],
            decoder_layers=config_param['decoder_layers'],
            decoder_heads=config_param['decoder_heads'],
            decoder_ff_dim=config_param['decoder_ff_dim'],
            decoder_dropout=config_param['decoder_dropout'],
            fp_dim=config_param['fp_dim'],
            target_var=config_param['target_var'],
        )

def get_criterion_name(criterion):
    """
    Get the name of a criterion (function or class).

    Args:
        criterion: The criterion object (function or class).

    Returns:
        str: The name of the criterion.
    """
    if criterion.__class__.__name__ == 'function':
        return criterion.__name__ 
    else:
        return criterion.__class__.__name__ 

class LatentSampler(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        LatentSampler: Encodes input into a latent space using mean and log variance.

        Args:
            input_dim (int): Dimension of the input features.
            latent_dim (int): Dimension of the latent space.
        """
        super(LatentSampler, self).__init__()

        # Linear layers for mean and log variance
        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_log_var = nn.Linear(input_dim, latent_dim)

    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick to sample z.

        Args:
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.

        Returns:
            torch.Tensor: Sampled latent variable z.
        """
        std = torch.exp(0.5 * log_var)  # Standard deviation
        epsilon = torch.randn_like(std)  # Random noise
        return mean + epsilon * std

    def forward(self, encoder_output):
        """
        Forward pass for LatentSampler.

        Args:
            encoder_output (torch.Tensor): Output from the encoder, shape (batch_size, input_dim).

        Returns:
            z (torch.Tensor): Sampled latent variable, shape (batch_size, latent_dim).
            mean (torch.Tensor): Mean of the latent variable, shape (batch_size, latent_dim).
            log_var (torch.Tensor): Log variance of the latent variable, shape (batch_size, latent_dim).
        """
        mean = self.fc_mean(encoder_output)  # Compute mean
        log_var = self.fc_log_var(encoder_output)  # Compute log variance

        z = self.reparameterize(mean, log_var)  # Sample z using the reparameterization trick

        return z, mean, log_var
    
    def calc_kl_divergence(self, mean, log_var, target_mean=None, target_var=1.0):
        """
        Calculate the KL divergence with optional target mean and variance.

        Args:
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.
            target_mean (float or None, optional): Target mean for the KL divergence.
                                                If None, the mean is unrestricted. Default is None.
            target_var (float, optional): Target variance for the KL divergence. Default is 1.0.

        Returns:
            torch.Tensor: KL divergence.
        """
        # Convert target variance to log variance
        target_log_var = torch.log(torch.tensor(target_var))

        if target_mean is not None:
            # KL divergence with mean and variance restrictions
            kl_divergence = -0.5 * torch.sum(
                1 + log_var - target_log_var - ((mean - target_mean).pow(2) + log_var.exp()) / target_var,
                dim=1
            )
        else:
            # KL divergence with variance restriction only
            kl_divergence = -0.5 * torch.sum(
                1 + log_var - target_log_var - log_var.exp() / target_var,
                dim=1
            )
        
        return kl_divergence.mean()




