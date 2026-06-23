import torch
import torch.nn as nn
import torch.nn.functional as F

class D4GroupAction:
    """
    Implements the Dihedral Group D_4 actions (8 symmetries: 4 rotations & 4 reflections)
    for 2D grid tensors of shape (Batch, Channel, Height, Width).
    """

    @staticmethod
    def apply_action(x, action_idx):
        """
        Applies one of the 8 group actions of D_4 on a grid tensor.
        action_idx range: 0 to 7.
        """
        # Symmetries:
        # 0: identity, 1: rot90, 2: rot180, 3: rot270
        # 4: flip, 5: flip + rot90, 6: flip + rot180, 7: flip + rot270
        rot_k = action_idx % 4
        flip = action_idx // 4

        out = x
        if flip == 1:
            out = torch.flip(out, dims=[-1])  # horizontal flip
        if rot_k > 0:
            out = torch.rot90(out, k=rot_k, dims=[-2, -1])
        return out

    @staticmethod
    def apply_inverse_action(x, action_idx):
        """
        Applies the mathematical inverse group action to realign features to original orientation.
        """
        rot_k = action_idx % 4
        flip = action_idx // 4

        out = x
        if rot_k > 0:
            out = torch.rot90(out, k=4 - rot_k, dims=[-2, -1])
        if flip == 1:
            out = torch.flip(out, dims=[-1])
        return out

    @staticmethod
    def get_action_permutation(action_idx):
        """
        Returns the permutation of the 8 action indices under group action action_idx.
        If we rotate/flip the grid, action `a` at state `s` corresponds to action `perm[a]` at state `g * s`.
        """
        action_vectors = [
            (-1, 0),  # 0: Up
            (1, 0),   # 1: Down
            (0, -1),  # 2: Left
            (0, 1),   # 3: Right
            (-1, -1), # 4: Up-Left
            (-1, 1),  # 5: Up-Right
            (1, -1),  # 6: Down-Left
            (1, 1)    # 7: Down-Right
        ]
        rot_k = action_idx % 4
        flip = action_idx // 4
        
        perm = []
        for dr, dc in action_vectors:
            if flip == 1:
                dc = -dc
            # rot90 CCW: dr_new = -dc, dc_new = dr
            for _ in range(rot_k):
                dr, dc = -dc, dr
            
            found = False
            for idx, vec in enumerate(action_vectors):
                if vec == (dr, dc):
                    perm.append(idx)
                    found = True
                    break
            if not found:
                raise ValueError(f"Vector ({dr}, {dc}) not found")
        return perm


class EquivariantConv2d(nn.Module):
    """
    D_4 Equivariant Convolutional Layer.
    Applies standard Conv2d over all 8 group-transformed views of the input,
    realigns the resulting feature maps using inverse transforms, and averages them.
    Guarantees: f(g * x) = g * f(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        # Share weights across all 8 symmetric views
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        batch_size, in_c, h, w = x.shape
        outputs = []

        # 1. Transform input under 8 symmetries, apply convolution, then inverse-transform back
        for i in range(8):
            transformed_input = D4GroupAction.apply_action(x, i)
            features = self.conv(transformed_input)
            realigned_features = D4GroupAction.apply_inverse_action(features, i)
            outputs.append(realigned_features)

        # 2. Aggregate by averaging (group projection)
        stacked_features = torch.stack(outputs, dim=0) # Shape: (8, Batch, OutChannels, H, W)
        equivariant_features = torch.mean(stacked_features, dim=0)
        return equivariant_features


class D4EquivariantNet(nn.Module):
    """
    Symmetric Actor-Critic Policy-Value Network using Group Frame Projection.
    Guarantees D4-equivariance for policy (8 directions) and D4-invariance for value.
    Inputs: (B, Channels, H, W)
    Outputs:
        - policy: (B, 8) probability distribution (Symmetric Action Space over 8 directions)
        - value: (B, 1) scalar value (Invariant under symmetries)
    """
    def __init__(self, board_size=13, in_channels=3, num_filters=64, num_layers=3):
        super().__init__()
        self.board_size = board_size
        self.base_net = StandardCNN(board_size, in_channels, num_filters, num_layers)
        # Precompute action permutations for all 8 group elements
        self.perms = [D4GroupAction.get_action_permutation(i) for i in range(8)]

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Transform input under 8 symmetries, stack into a single batch
        xs_trans = []
        for i in range(8):
            xs_trans.append(D4GroupAction.apply_action(x, i))
        x_batched = torch.cat(xs_trans, dim=0) # (8 * B, C, H, W)
        
        # 2. Forward pass through standard CNN
        logits_batched, values_batched = self.base_net(x_batched)
        
        # 3. Reshape outputs to separate group dimension
        logits_g = logits_batched.view(8, batch_size, 8)
        values_g = values_batched.view(8, batch_size, 1)
        
        # 4. Apply permutation to realign policy outputs to original orientation
        realigned_logits_list = []
        for i in range(8):
            logits_i = logits_g[i] # (B, 8)
            perm = self.perms[i]
            realigned_i = logits_i[:, perm]
            realigned_logits_list.append(realigned_i)
            
        # 5. Average policy and value outputs over the group (projection)
        policy_logits = torch.mean(torch.stack(realigned_logits_list, dim=0), dim=0)
        value = torch.mean(values_g, dim=0)
        
        return policy_logits, value


class StandardCNN(nn.Module):
    """
    Standard CNN with equivalent depth and layer configurations,
    but without equivariant constraints. Outputs 8 action logits.
    """
    def __init__(self, board_size=13, in_channels=3, num_filters=64, num_layers=3):
        super().__init__()
        self.board_size = board_size

        # Stack standard convolutional blocks
        layers = [
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU()
            ]
        self.backbone = nn.Sequential(*layers)

        # Policy Head (Outputs 8 directions)
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, 8)
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value
