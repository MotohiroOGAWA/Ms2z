
        # self.atom_counter_tensors = nn.Parameter(self.vocab.atom_counter_tensors, requires_grad=False)
        # self.inner_bond_counter_tensors = nn.Parameter(self.vocab.inner_bond_counter_tensors, requires_grad=False)
        # self.outer_bond_cnt_tensors = nn.Parameter(self.vocab.outer_bond_cnt_tensors, requires_grad=False)



# class MolFeatureLoss(nn.Module):
#     def __init__(self, vocab: Vocab, atom_counter_tensors, inner_bond_counter_tensors, outer_bond_cnt_tensors):
#         super(MolFeatureLoss, self).__init__()
#         self.vocab = vocab
#         self.atom_counter_tensors = atom_counter_tensors
#         self.inner_bond_counter_tensors = inner_bond_counter_tensors
#         self.outer_bond_cnt_tensors = outer_bond_cnt_tensors

#     def forward(self, predicted_tokens, target_tokens, latent_mean, latent_log_var):
#         """
#         Calculate the total loss for the model.

#         Args:
#             predicted_tokens (torch.Tensor): Predicted token indices, shape [batch_size, seq_len].
#             target_tokens (torch.Tensor): Ground truth token indices, shape [batch_size, seq_len].
#             latent_mean (torch.Tensor): Mean of the latent distribution, shape [batch_size, latent_dim].
#             latent_log_var (torch.Tensor): Log variance of the latent distribution, shape [batch_size, latent_dim].

#         Returns:
#             tuple: Loss components (token_loss, atom_loss, bond_loss, external_bond_loss, kl_divergence_loss).
#         """
#         # Identify special tokens for BOS and EOS
#         special_tokens = torch.tensor([self.vocab.bos, self.vocab.eos], device=predicted_tokens.device)

#         # Move atom, bond, and external bond tensors to the same device as predictions
#         self.vocab.atom_counter_tensors = self.vocab.atom_counter_tensors.to(predicted_tokens.device)
#         self.vocab.inner_bond_counter_tensors = self.vocab.inner_bond_counter_tensors.to(predicted_tokens.device)
#         self.vocab.outer_bond_cnt_tensors = self.vocab.outer_bond_cnt_tensors.to(predicted_tokens.device)

#         # Flatten the predicted and target tensors for processing
#         predicted_flat = predicted_tokens.view(-1)
#         target_flat = target_tokens.view(-1)

#         # Create a mask to exclude padding tokens
#         valid_mask = target_flat != self.vocab.pad

#         # Filter out padding tokens
#         predicted_flat = predicted_flat[valid_mask]
#         target_flat = target_flat[valid_mask]

#         # Separate special tokens (BOS and EOS) for individual processing
#         special_token_mask = torch.isin(target_flat, special_tokens)

#         predicted_special_tokens = predicted_flat[special_token_mask]
#         target_special_tokens = target_flat[special_token_mask]
#         predicted_main_tokens = predicted_flat[~special_token_mask]
#         target_main_tokens = target_flat[~special_token_mask]

#         # Calculate token-level mismatch for special tokens
#         special_token_mismatch = torch.sum(predicted_special_tokens != target_special_tokens).float()

#         # Exclude any BOS/EOS tokens from the main tokens
#         main_token_mask = ~torch.isin(predicted_main_tokens, special_tokens)
#         predicted_main_tokens = predicted_main_tokens[main_token_mask]
#         target_main_tokens = target_main_tokens[main_token_mask]

#         # Atom counter similarity
#         predicted_atom_features = self.vocab.atom_counter_tensors[predicted_main_tokens]
#         target_atom_features = self.vocab.atom_counter_tensors[target_main_tokens]
#         atom_similarity = F.cosine_similarity(predicted_atom_features, target_atom_features, dim=1)

#         # Inner bond counter similarity
#         predicted_bond_features = self.vocab.inner_bond_counter_tensors[predicted_main_tokens]
#         target_bond_features = self.vocab.inner_bond_counter_tensors[target_main_tokens]
#         bond_similarity = F.cosine_similarity(predicted_bond_features, target_bond_features, dim=1)

#         # Outer bond count differences
#         predicted_external_bond_counts = self.vocab.outer_bond_cnt_tensors[predicted_main_tokens]
#         target_external_bond_counts = self.vocab.outer_bond_cnt_tensors[target_main_tokens]
#         external_bond_difference = self.outer_bond_similarity(predicted_external_bond_counts, target_external_bond_counts)

#         # Calculate loss components
#         token_mismatch_loss = (special_token_mismatch / max(1, target_special_tokens.size(0))).to(predicted_tokens.device)
#         atom_loss = atom_similarity.mean().to(predicted_tokens.device)
#         bond_loss = bond_similarity.mean().to(predicted_tokens.device)
#         external_bond_loss = external_bond_difference.mean().to(predicted_tokens.device)

#         # KL Divergence loss
#         kl_divergence_loss = -0.5 * torch.sum(1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp())
#         kl_divergence_loss /= latent_mean.size(0)  # Normalize by batch size

#         return token_mismatch_loss, atom_loss, bond_loss, external_bond_loss, kl_divergence_loss

class MolFeatureLoss(nn.Module):
    def __init__(self, vocab: Vocab, atom_counter_tensors, inner_bond_counter_tensors, outer_bond_cnt_tensors):
        super(MolFeatureLoss, self).__init__()
        self.vocab = vocab
        self.atom_counter_tensors = atom_counter_tensors
        self.inner_bond_counter_tensors = inner_bond_counter_tensors
        self.outer_bond_cnt_tensors = outer_bond_cnt_tensors

    def forward(self, predicted_probabilities, target_tokens, latent_mean, latent_log_var):
        """
        Calculate the total loss for the model.

        Args:
            predicted_probabilities (torch.Tensor): Predicted probabilities for each token, shape [batch_size, seq_len, vocab_size].
            target_tokens (torch.Tensor): Ground truth token indices, shape [batch_size, seq_len].
            latent_mean (torch.Tensor): Mean of the latent distribution, shape [batch_size, latent_dim].
            latent_log_var (torch.Tensor): Log variance of the latent distribution, shape [batch_size, latent_dim].

        Returns:
            tuple: Loss components (token_loss, atom_loss, bond_loss, external_bond_loss, kl_divergence_loss).
        """
        # Flatten probabilities and target tokens
        batch_size, seq_len, vocab_size = predicted_probabilities.size()
        predicted_flat = predicted_probabilities.view(-1, vocab_size)
        target_flat = target_tokens.view(-1)

        mask = target_flat != self.vocab.pad
        predicted_flat = predicted_flat[mask]
        target_flat = target_flat[mask]

        # Cross-entropy loss for tokens
        token_mismatch_loss = nn.CrossEntropyLoss(ignore_index=self.vocab.pad)(
            predicted_flat, target_flat
        )

        # Atom counter similarity
        predicted_atom_features = self.atom_counter_tensors[predicted_flat]  # [batch_size * seq_len, feature_dim]
        target_atom_features = self.atom_counter_tensors[target_flat]
        atom_similarity = F.cosine_similarity(predicted_atom_features, target_atom_features, dim=-1)

        # Inner bond counter similarity
        predicted_bond_features = torch.matmul(predicted_probabilities, self.inner_bond_counter_tensors)
        target_bond_features = self.inner_bond_counter_tensors[target_flat]
        bond_similarity = F.cosine_similarity(predicted_bond_features, target_bond_features, dim=-1)

        # Outer bond count differences
        predicted_external_bond_counts = torch.matmul(predicted_probabilities, self.outer_bond_cnt_tensors)
        target_external_bond_counts = self.outer_bond_cnt_tensors[target_flat]
        external_bond_difference = self.outer_bond_similarity(predicted_external_bond_counts, target_external_bond_counts)

        # KL Divergence loss
        kl_divergence_loss = -0.5 * torch.sum(1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp())
        kl_divergence_loss /= latent_mean.size(0)  # Normalize by batch size

        # Return loss components
        return token_mismatch_loss, atom_similarity.mean(), bond_similarity.mean(), external_bond_difference.mean(), kl_divergence_loss

    def outer_bond_similarity(self, predicted_counts, target_counts, epsilon=1e-6):
        """
        Calculate similarity for outer bond counts.

        Args:
            predicted_counts (torch.Tensor): Predicted outer bond counts, shape [N].
            target_counts (torch.Tensor): Target outer bond counts, shape [N].
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            torch.Tensor: Similarity scores for each pair, shape [N].
        """
        # Compute absolute differences
        differences = torch.abs(predicted_counts - target_counts)

        # Compute element-wise maximum to normalize
        max_values = torch.max(predicted_counts, target_counts) + epsilon

        # Normalize the differences
        normalized_difference = differences / max_values

        return normalized_difference
