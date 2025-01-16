
import torch
import torch.nn as nn

from .graph_lib import *
from .chem_encoder import *
from .decoder import *

class Ms2z(nn.Module):
    def __init__(
            self, vocab_data, max_seq_len, node_dim, edge_dim, joint_edge_dim,
            latent_dim,
            decoder_layers, decoder_heads, decoder_ff_dim, decoder_dropout,
            fp_dim,
            target_var=1.0,
            tgt_token_priority_weight=0.2,
            ):
        super(Ms2z, self).__init__()
        vocab_size = len(vocab_data['token'])
        bos_idx = vocab_data['bos']
        eos_idx = vocab_data['eos']
        pad_idx = vocab_data['pad']
        unk_idx = vocab_data['unk']
        joint_potential = vocab_data['joint_potential']
        fp_tensor = vocab_data['fingerprint']
        formula_tensor = vocab_data['formula']
        # tgt_similar = vocab_data['tgt_cosine_matrix']


        self.vocab_size = vocab_size

        self.max_seq_len = max_seq_len
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.joint_edge_dim = joint_edge_dim
        self.latent_dim = latent_dim
        self.joint_potential_size = joint_potential.size(1)

        self.tgt_token_priority_weight = tgt_token_priority_weight

        # Vocabulary embedding
        self.vocab_embedding = FragEmbeddings(
            node_dim, edge_dim, joint_edge_dim,
            self.vocab_size,
            joint_potential,
            fp_tensor,
            formula_tensor,
            bos_idx, eos_idx, pad_idx, unk_idx,
            )
        self.bos = nn.Parameter(torch.tensor(bos_idx, dtype=torch.int32), requires_grad=False)
        self.eos = nn.Parameter(torch.tensor(eos_idx, dtype=torch.int32), requires_grad=False)
        self.pad = nn.Parameter(torch.tensor(pad_idx, dtype=torch.int32), requires_grad=False)
        self.unk = nn.Parameter(torch.tensor(unk_idx, dtype=torch.int32), requires_grad=False)
        # self.tgt_similar = nn.Parameter(tgt_similar, requires_grad=False)

        # Structure encoder
        encoder_h_size = 4*node_dim
        self.chem_encoder = StructureEncoder(node_dim=node_dim, edge_dim=edge_dim, h_size=encoder_h_size)

        # MS encoder
        self.ms_encoder = None

        # Latent sampler
        self.chem_latent_sampler = LatentSampler(input_dim=encoder_h_size, latent_dim=latent_dim)
        self.memory_linear = nn.Linear(latent_dim, node_dim+edge_dim+joint_edge_dim)
        self.target_var = target_var

        # Transformer decoder
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.decoder_ff_dim = decoder_ff_dim
        self.decoder_dropout = decoder_dropout
        self.decoder = Decoder(decoder_layers, node_dim+edge_dim+joint_edge_dim, self.joint_potential_size, decoder_heads, decoder_ff_dim, decoder_dropout)
        
        # loss_fn2 = nn.CrossEntropyLoss(ignore_index=-1)
        # self.similar_token_loss_fn = F.binary_cross_entropy_with_logits
        # self.output_joint_layer = nn.Linear(node_dim, self.vocab_embedding.max_joint_cnt)

        # predict Fingerprint from z
        self.fp_dim = fp_dim
        self.z_fp_layer = nn.Sequential(
            nn.Linear(latent_dim, 2*fp_dim),
            nn.ReLU(),
            nn.Linear(2*fp_dim, fp_dim),
            nn.Dropout(0.1),
        )
        self.z_fp_loss_fn = nn.MSELoss(reduction='none')
        
        # predict token
        self.output_word_layer = nn.Linear(node_dim, self.vocab_size)
        # self.pred_token_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='none')
        self.pred_token_loss_fn = F.cross_entropy

        # predict root joint
        self.root_joint_layer = nn.Sequential(
            nn.Linear(edge_dim, self.joint_potential_size),
            nn.ReLU(),
            nn.Linear(self.joint_potential_size, self.joint_potential_size),
            nn.Dropout(0.1),
        )
        self.pred_root_joint_loss_fn = F.cross_entropy

        # predict parent joint
        self.parent_joint_linear = nn.Linear(joint_edge_dim, joint_edge_dim)
        self.parent_joint_pos_layer = nn.Sequential(
            nn.Linear(joint_edge_dim, joint_edge_dim),
            nn.ReLU(),
            nn.Linear(joint_edge_dim, self.joint_potential_size),
        )
        self.parent_bond_type_layer = nn.Sequential(
            nn.Linear(joint_edge_dim, joint_edge_dim),
            nn.ReLU(),
            nn.Linear(joint_edge_dim, 3),
        )
        self.pred_parent_joint_loss_fn = F.cross_entropy


        # Vocab embedding loss
        self.vocab_embed_linear = nn.Linear(node_dim, node_dim)

        # self.fp_linear1 = nn.Linear(node_dim+edge_dim, 512)
        # self.fp_linear2 = nn.Linear(512, 167)

    def forward(self, vocab_tensor, order_tensor, mask_tensor, fp_tensor):
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
        # # Update vocab_tensor using mask_tensor to insert eos and pad tokens
        # token_seq_tensor = torch.cat([
        #     torch.full((vocab_tensor.size(0), 1), self.bos, dtype=vocab_tensor.dtype, device=vocab_tensor.device), 
        #     vocab_tensor], dim=1)
        # mask_tensor_ex = torch.cat([torch.ones(mask_tensor.size(0),1, dtype=torch.bool, device=mask_tensor.device), mask_tensor], dim=1)
        # target_tensor = token_seq_tensor.roll(shifts=-1, dims=1).long()
        # parent_idx_tensor = order_tensor[:,:,0] + 1
        # parent_token_seq_tensor = torch.gather(token_seq_tensor[:,:-1], dim=1, index=parent_idx_tensor)
        # parent_token_seq_tensor = torch.cat([torch.full((parent_token_seq_tensor.size(0), 1), -1, dtype=parent_token_seq_tensor.dtype, device=parent_token_seq_tensor.device), parent_token_seq_tensor], dim=1)
        # for i in range(token_seq_tensor.size(0)):  # Iterate over batch
        #     eos_index = torch.where(mask_tensor_ex[i] == False)[0]
        #     if eos_index.numel() > 0:  # If there is at least one False in mask_tensor
        #         first_eos_index = eos_index[0].item()
        #         target_tensor[i, first_eos_index-1] = self.eos 
        #         target_tensor[i, first_eos_index:] = self.pad  
        #         token_seq_tensor[i, first_eos_index:] = self.pad
        #         parent_token_seq_tensor[i, first_eos_index:] = -1

        # # Process input sequence through the decoder
        # padding = torch.empty(mask_tensor.size(0), 1, device=mask_tensor.device, dtype=torch.int64)
        # padding.fill_(-1)
        # edge_index = torch.where(
        #     mask_tensor,
        #     order_tensor[:,:,0],  # Retain original values where mask_tensor is True
        #     torch.tensor(-2, dtype=torch.int64)  # Replace with self.vocab.pad where mask_tensor is False
        # )
        # edge_index = torch.cat([padding,edge_index+1], dim=1) # [batch_size, seq_len]
        # root_atom_pos_tensor = torch.cat([padding, order_tensor[:,:,2]], dim=1)
        # tgt_root_atom_pos_tensor = torch.cat([order_tensor[:,:,2], padding], dim=1)
        # joint_atom_pos_tensor = torch.cat([order_tensor[:,:,1], padding], dim=1)
        # tgt_joint_atom_pos_tensor = torch.cat([order_tensor[:,:,1], padding], dim=1)
        # root_bond_type_tensor = torch.cat([padding, order_tensor[:,:,3]], dim=1)
        # tgt_root_bond_type_tensor = torch.cat([order_tensor[:,:,3], padding], dim=1)
        # joint_bond_type_tensor = torch.cat([order_tensor[:,:,3], padding], dim=1)
        # root_atom_pos_and_type_tensor = torch.stack([root_atom_pos_tensor, root_bond_type_tensor], dim=-1)
        # joint_atom_pos_and_type_tensor = torch.stack([joint_atom_pos_tensor, joint_bond_type_tensor], dim=-1)

        # node_embed = self.vocab_embedding(token_seq_tensor, root_atom_pos_and_type_tensor)  # Embed vocabulary tensor [batch_size, seq_len+1, node_dim + edge_dim]
        
        # token_and_joint_tensor = torch.cat([token_seq_tensor.unsqueeze(-1), joint_atom_pos_and_type_tensor], dim=-1) # [batch_size, seq_len+1, 3]
        # edge_attr = self.vocab_embedding.edge_embed(token_and_joint_tensor) # [batch_size, seq_len+1, edge_dim]
        # unk_nodes = self.vocab_embedding(self.unk) # [node_dim+edge_dim]
        # unk_nodes = torch.cat([unk_nodes, torch.zeros(self.joint_edge_dim, dtype=unk_nodes.dtype, device=unk_nodes.device)], dim=-1) # [node_dim+edge_dim+joint_edge_dim]

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
        node_joint_potentials = self.vocab_embedding.joint_potential[parent_token_seq_tensor][...,1]

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
        
        # Fingerprint loss
        z_fp_loss = self.calc_fp_loss(z, fp_tensor)

        # KL divergence loss
        kl_divergence_loss = self.chem_latent_sampler.calc_kl_divergence(mean, log_var, target_var=self.target_var)

        # calc token loss
        predict_token_loss, predict_token_acc = self.calc_predict_token_loss(node_embed_flat, tgt_token_id_flat)

        # calc root joint loss
        predict_root_joint_loss, predict_root_joint_acc = self.calc_predict_root_joint_loss(root_edge_embed_flat, tgt_root_atom_pos_flat)
        
        # calc parent joint loss
        predict_parent_joint_loss, predict_parent_joint_acc = self.calc_precdict_parent_joint_loss(parent_joint_edge_embed_flat, tgt_parent_atom_pos_flat, tgt_bond_type_flat)

        # vocab embedding loss
        ve_fp_loss, ve_formula_loss, ve_special_tokens_loss = self.calc_node_embed_loss(node_embed_flat, tgt_token_id_flat)
        ve_loss = ve_fp_loss + ve_formula_loss + ve_special_tokens_loss

        # vocab embedding loss direct
        unique_token_indices = torch.unique(tgt_token_id_flat)
        unique_token_indices = unique_token_indices[unique_token_indices != self.eos]
        token_fp_loss, token_formula_loss, token_special_tokens_loss = self.vocab_embedding.train_batch_nodes(unique_token_indices)
        token_loss = token_fp_loss + token_formula_loss + token_special_tokens_loss

        loss_list = {
            'z_fp': z_fp_loss,
            'KL': kl_divergence_loss,
            'pred_token': predict_token_loss,
            'pred_root_joint': predict_root_joint_loss,
            'pred_parent_joint': predict_parent_joint_loss,
            've': ve_loss,
            'token': token_loss,
        }
        acc_list = {
            'pred_token': predict_token_acc.item(),
        }
        target_data = {
            'z_fp': {'loss': z_fp_loss.item(), 'accuracy': None, 'criterion': get_criterion_name(self.z_fp_loss_fn)},
            'KL': {'loss': kl_divergence_loss.item(), 'accuracy': None, 'criterion': 'KL Divergence'},
            'pred_token': {'loss': predict_token_loss.item(), 'accuracy': predict_token_acc.item(), 'criterion': get_criterion_name(self.pred_token_loss_fn)},
            'pred_root_joint': {'loss': predict_root_joint_loss.item(), 'accuracy': predict_root_joint_acc.item(), 'criterion': get_criterion_name(self.pred_root_joint_loss_fn)},
            'pred_parent_joint': {'loss': predict_parent_joint_loss.item(), 'accuracy': predict_parent_joint_acc.item(), 'criterion': get_criterion_name(self.pred_parent_joint_loss_fn)},
            # 've': {'loss': ve_loss.item(), 'accuracy': None, 'criterion': get_criterion_name(self.vocab_embedding.fp_loss_fn)},
            've_fp': {'loss': ve_fp_loss.item(), 'accuracy': None, 'criterion': get_criterion_name(self.vocab_embedding.fp_loss_fn)},
            've_F': {'loss': ve_formula_loss.item(), 'accuracy': None, 'criterion': get_criterion_name(self.vocab_embedding.formula_loss_fn)},
            've_spe': {'loss': ve_special_tokens_loss.item(), 'accuracy': None, 'criterion': get_criterion_name(self.vocab_embedding.special_tokens_loss_fn)},
            'token': {'loss': token_loss.item(), 'accuracy': None, 'criterion': get_criterion_name(self.vocab_embedding.fp_loss_fn)},
        }

        return loss_list, acc_list, target_data
    
    def prepare_input(self, vocab_tensor, order_tensor, mask_tensor):
        """
        Prepare input for forward pass.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. (parent_idx, parent_bond_pos, bond_pos, parent_atom_idx, atom_idx, bond_type_num)
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            tuple: Input tuple for forward pass.
        """
        # Update vocab_tensor using mask_tensor to insert eos and pad tokens
        token_seq_tensor = torch.cat([
            torch.full((vocab_tensor.size(0), 1), self.bos, dtype=vocab_tensor.dtype, device=vocab_tensor.device), 
            vocab_tensor], dim=1)
        mask_tensor_ex = torch.cat([torch.ones(mask_tensor.size(0),1, dtype=torch.bool, device=mask_tensor.device), mask_tensor], dim=1)
        target_tensor = token_seq_tensor.roll(shifts=-1, dims=1).long()
        parent_idx_tensor = order_tensor[:,:,0] + 1
        parent_token_seq_tensor = torch.gather(token_seq_tensor[:,:-1], dim=1, index=parent_idx_tensor)
        parent_token_seq_tensor = torch.cat([torch.full((parent_token_seq_tensor.size(0), 1), -1, dtype=parent_token_seq_tensor.dtype, device=parent_token_seq_tensor.device), parent_token_seq_tensor], dim=1)
        for i in range(token_seq_tensor.size(0)):  # Iterate over batch
            eos_index = torch.where(mask_tensor_ex[i] == False)[0]
            if eos_index.numel() > 0:  # If there is at least one False in mask_tensor
                first_eos_index = eos_index[0].item()
                target_tensor[i, first_eos_index-1] = self.eos 
                target_tensor[i, first_eos_index:] = self.pad  
                token_seq_tensor[i, first_eos_index:] = self.pad
                parent_token_seq_tensor[i, first_eos_index:] = -1

        # Process input sequence through the decoder
        padding = torch.empty(mask_tensor.size(0), 1, device=mask_tensor.device, dtype=torch.int64)
        padding.fill_(-1)
        edge_index = torch.where(
            mask_tensor,
            order_tensor[:,:,0],  # Retain original values where mask_tensor is True
            torch.tensor(-2, dtype=torch.int64)  # Replace with self.vocab.pad where mask_tensor is False
        )
        edge_index = torch.cat([padding,edge_index+1], dim=1) # [batch_size, seq_len]
        root_atom_pos_tensor = torch.cat([padding, order_tensor[:,:,2]], dim=1)
        tgt_root_atom_pos_tensor = torch.cat([order_tensor[:,:,2], padding], dim=1)
        joint_atom_pos_tensor = torch.cat([order_tensor[:,:,1], padding], dim=1)
        tgt_joint_atom_pos_tensor = torch.cat([order_tensor[:,:,1], padding], dim=1)
        root_bond_type_tensor = torch.cat([padding, order_tensor[:,:,3]], dim=1)
        tgt_root_bond_type_tensor = torch.cat([order_tensor[:,:,3], padding], dim=1)
        joint_bond_type_tensor = torch.cat([order_tensor[:,:,3], padding], dim=1)
        root_atom_pos_and_type_tensor = torch.stack([root_atom_pos_tensor, root_bond_type_tensor], dim=-1)
        joint_atom_pos_and_type_tensor = torch.stack([joint_atom_pos_tensor, joint_bond_type_tensor], dim=-1)

        node_embed = self.vocab_embedding(token_seq_tensor, root_atom_pos_and_type_tensor)  # Embed vocabulary tensor [batch_size, seq_len+1, node_dim + edge_dim]
        
        token_and_joint_tensor = torch.cat([token_seq_tensor.unsqueeze(-1), joint_atom_pos_and_type_tensor], dim=-1) # [batch_size, seq_len+1, 3]
        edge_attr = self.vocab_embedding.edge_embed(token_and_joint_tensor) # [batch_size, seq_len+1, edge_dim]
        unk_nodes = self.vocab_embedding(self.unk) # [node_dim+edge_dim]
        unk_nodes = torch.cat([unk_nodes, torch.zeros(self.joint_edge_dim, dtype=unk_nodes.dtype, device=unk_nodes.device)], dim=-1) # [node_dim+edge_dim+joint_edge_dim]

        return node_embed, edge_attr, token_seq_tensor, edge_index, mask_tensor_ex, target_tensor, parent_token_seq_tensor, root_atom_pos_tensor, tgt_root_atom_pos_tensor, joint_atom_pos_tensor, tgt_joint_atom_pos_tensor, root_bond_type_tensor, tgt_root_bond_type_tensor, joint_bond_type_tensor, unk_nodes
    
    def translate_z(self, vocab_tensor, order_tensor, mask_tensor):
        """
        Translate latent variable z to a sequence.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. (parent_idx, parent_bond_pos, bond_pos, parent_atom_idx, atom_idx, bond_type_num)
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            torch.Tensor: Translated sequence.
        """
        # Calculate latent variable
        node_embed, edge_attr, token_seq_tensor, edge_index, mask_tensor_ex, \
            target_tensor, parent_token_seq_tensor, root_atom_pos_tensor, \
                tgt_root_atom_pos_tensor, joint_atom_pos_tensor, tgt_joint_atom_pos_tensor, \
                    root_bond_type_tensor, tgt_root_bond_type_tensor, joint_bond_type_tensor, \
                        unk_nodes = self.prepare_input(vocab_tensor, order_tensor, mask_tensor)
        z, mean, log_var = self.calc_chem_z(node_embed[:,1:], edge_attr[:,1:], order_tensor, mask_tensor)
        
        return mean

    def calc_chem_z(self, embed_node, edge_attr, order_tensor, mask_tensor):
        """
        Calculate the latent variable for a given input.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. (parent_idx, parent_bond_pos, bond_pos, parent_atom_idx, atom_idx, bond_type_num)
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
        """
        # Convert orders and masks into adjacency matrices
        adj_matrix_list = [order_to_adj_matrix(order[:, 0], mask) for order, mask in zip(order_tensor, mask_tensor)]

        # Pass through structure encoder
        encoder_output = self.chem_encoder(embed_node, edge_attr, adj_matrix_list, mask_tensor)

        # Sample from latent space
        z, mean, log_var = self.chem_latent_sampler(encoder_output)

        return z, mean, log_var
    
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

    def calc_predict_token_loss(self, node_embed, tgt_token_id, detail=False):
        # Project node embeddings to vocabulary logits
        logits_vocab = self.output_word_layer(node_embed)  # Shape: [data_size, node_dim] -> [data_size, vocab_size]
        # Apply Softmax to get probabilities
        # probabilities = F.softmax(logits_vocab, dim=-1)  # Shape: [data_size, vocab_size]
        
        losses = self.pred_token_loss_fn(logits_vocab, tgt_token_id, reduction='none')

        # Compute weights for each token
        unique_tokens, counts = torch.unique(tgt_token_id, return_counts=True)
        token_weights = (1 / counts.float()) / unique_tokens.size(0)

        # Create a weight tensor for each target
        weight_map = {token.item(): weight for token, weight in zip(unique_tokens, token_weights)}
        weights = torch.tensor([weight_map[token.item()] for token in tgt_token_id], device=node_embed.device)  # Shape: [data_size]

        # Apply weights to the individual losses
        weighted_losses = losses * weights  # Shape: [data_size]

        # Compute the weighted average loss
        loss = weighted_losses.sum()

        # Compute the accuracy
        predicted_token_id = torch.argmax(logits_vocab, dim=1)
        correct = (predicted_token_id == tgt_token_id).float()
        accuracy = correct.mean()
        # tgt_token_id[torch.where(correct >= 1.0)[0]]

        if detail:
            return losses, correct
        else:
            return loss, accuracy

    def calc_predict_root_joint_loss(self, edge_embed, tgt_root_joint):
        if edge_embed.size(0) == 0:
            return torch.tensor(0.0, device=edge_embed.device), torch.tensor(0.0, device=edge_embed.device)
        # Project node embeddings to vocabulary logits
        logits_root_joint = self.root_joint_layer(edge_embed)  # Shape: [data_size, edge_dim] -> [data_size, poteintial_joint_size]
        loss = self.pred_root_joint_loss_fn(logits_root_joint, tgt_root_joint)

        # Compute the accuracy
        predicted_root_joint = torch.argmax(logits_root_joint, dim=1)
        correct = (predicted_root_joint == tgt_root_joint).float()
        accuracy = correct.mean()

        return loss, accuracy
    
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
        return self.vocab_embedding.calc_node_loss(node_embed, tgt_token_id_flat)

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
        node_joint_potentials = self.vocab_embedding.joint_potential[parent_token_seq_tensor][...,1]

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




