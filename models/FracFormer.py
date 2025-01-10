
import torch
from torch import nn, einsum
from models.FAPE import DGCNN_FAPE
from timm.models.layers import DropPath, trunc_normal_
import argparse
from knn_cuda import KNN
from LossFunctions.chamfer_dist import ChamferDistanceL1,ChamferDistanceL2

knn_frac = KNN(k=8, transpose_mode=False)

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def get_knn_index(coor_q, coor_k=None):
    coor_k = coor_k if coor_k is not None else coor_q
    # coor: bs, 3, np
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        #         _, idx = knn(coor_k, coor_q)  # bs k np
        idx = knn_point(8, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous())  # B G M
        idx = idx.transpose(-1, -2).contiguous()
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)

    return idx  # bs*k*np


def get_knn_index_inter(coor_q, coor_k=None):
    k = 8
    coor_k = coor_k if coor_k is not None else coor_q
    batch_size, _, num_points_q = coor_q.shape
    num_points_k = coor_k.shape[2]
    with torch.no_grad():
        # Assume labels_q and labels_k are the last five dimensions of coor_q and coor_k (5 for one-hot dimension)
        labels_q = coor_q[:, 3:, :]  # [B, 5, npoints_q]
        labels_k = coor_k[:, 3:, :]  # [B, 5, npoints_k]

        # Initialize feature matrices for same-label and different-label neighbors
        all_neighbors_indices = torch.full((batch_size, num_points_q, k), -1, dtype=torch.int64, device=coor_q.device)
        all_neighbors_indices_other = torch.full((batch_size, num_points_q, k), -1, dtype=torch.int64, device=coor_q.device)

        for b in range(batch_size):
            # Identify active labels (labels that have points)
            active_labels = torch.any(labels_q[b].bool(), dim=1)

            for label in torch.where(active_labels)[0]:  # Iterate over labels with points
                mask_q = labels_q[b, label, :].bool()
                mask_k = labels_k[b, label, :].bool()

                coor_q_label = coor_q[b, :3, mask_q]
                coor_k_label = coor_k[b, :3, mask_k]

                # KNN for same-label points
                if mask_k.sum() >= k:
                    _, idx_knn_same = knn_frac(coor_k_label.unsqueeze(0), coor_q_label.unsqueeze(0))
                    idx_knn_same = idx_knn_same.squeeze(0)
                else:
                    repeat_times = (k // coor_k_label.size(1)) + 1
                    idx_knn_same = torch.arange(coor_k_label.size(1), device=coor_k_label.device).repeat(repeat_times)[:k]
                    idx_knn_same = idx_knn_same.unsqueeze(1).repeat(1, coor_q_label.size(1))

                extracted_indices_same = torch.where(mask_k)[0][idx_knn_same]
                all_neighbors_indices[b, mask_q, :] = extracted_indices_same.permute(1, 0)

                # KNN for different-label points
                mask_k_other = ~mask_k & torch.any(labels_k[b].bool(), dim=0)  # Select different-label points
                coor_k_other = coor_k[b, :3, mask_k_other]

                if mask_k_other.sum() >= k:
                    _, idx_knn_other = knn_frac(coor_k_other.unsqueeze(0), coor_q_label.unsqueeze(0))
                    idx_knn_other = idx_knn_other.squeeze(0)
                else:
                    repeat_times = (k // coor_k_other.size(1)) + 1
                    idx_knn_other = torch.arange(coor_k_other.size(1), device=coor_k_other.device).repeat(repeat_times)[:k]
                    idx_knn_other = idx_knn_other.unsqueeze(1).repeat(1, coor_q_label.size(1))

                extracted_indices_other = torch.where(mask_k_other)[0][idx_knn_other]
                all_neighbors_indices_other[b, mask_q, :] = extracted_indices_other.permute(1, 0)

        # Flatten indices for compatibility with the output format of `get_knn_index`
        idx_same = all_neighbors_indices.view(batch_size, -1)  # Flatten to [B, np*k]
        idx_other = all_neighbors_indices_other.view(batch_size, -1)  # Flatten to [B, np*k]

        # Adjust index with batch offset
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1) * num_points_k
        idx_same += idx_base
        idx_other += idx_base

        # Flatten to 1D tensor
        idx_same = idx_same.view(-1)
        idx_other = idx_other.view(-1)

    return idx_same, idx_other  # Both indices in flattened format for same and different labels

def get_graph_feature(x, knn_index, x_q=None):
    # x: bs, np, c, knn_index: bs*k*np
    k = 8
    batch_size, num_points, num_dims = x.size()
    num_query = x_q.size(1) if x_q is not None else num_points
    feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
    feature = feature.view(batch_size, k, num_query, num_dims)
    x = x_q if x_q is not None else x
    x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
    feature = torch.cat((feature - x, x), dim=-1)
    return feature  # b k np c

# rebuild head
class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, self.step*3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature

        patch_feature = torch.cat([
            g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
            token_feature
        ], dim=-1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step, 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc


# Basic module
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EncoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map_intra = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_map = nn.Linear(dim * 2, dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_index_intra=None):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)
        if knn_index_intra is not None:
            knn_f_intra = get_graph_feature(norm_x, knn_index_intra)
            knn_f_intra = self.knn_map_intra(knn_f_intra)
            knn_f_intra = knn_f_intra.max(dim=1, keepdim=False)[0]

            x_1 = torch.cat([x_1, knn_f_intra], dim=-1)

            x_1 = self.merge_map(x_1)

        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim * 2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(dim * 2, dim)

    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        if self_knn_index is not None:
            knn_f = get_graph_feature(norm_q, self_knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_1 = torch.cat([q_1, knn_f], dim=-1)
            q_1 = self.merge_map(q_1)

        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        if cross_knn_index is not None:
            knn_f = get_graph_feature(norm_v, cross_knn_index, norm_q)
            knn_f = self.knn_map_cross(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = self.merge_map_cross(q_2)

        q = q + self.drop_path(q_2)

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q

# models Arc
class Fracformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        # params
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        self.num_points = None
        self.num_features = self.embed_dim = config.trans_dim
        self.One_hot_dim = config.One_hot_dim
        self.in_chans = 3
        self.depth = [6,8]

        self.fold_step = int(pow(self.num_pred // self.num_query, 0.5) + 0.5)
        # Frac-wise encoder
        self.FAPE = DGCNN_FAPE([self.num_query*2, self.num_query])
        # Embeding
        self.pos_embed = nn.Sequential(
            nn.Conv1d(self.in_chans+self.One_hot_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, self.embed_dim, 1)
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(128, self.embed_dim, 1),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.embed_dim, self.embed_dim, 1)
        )

        # Transformer encoder
        self.encoder = nn.ModuleList([
            EncoderBlock(dim=self.embed_dim, num_heads=6, mlp_ratio=2., qkv_bias=False, qk_scale=None,drop=0., attn_drop=0.)
            for i in range(self.depth[0])])

        # Patch moving
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, self.embed_dim, 1)
        )
        self.increase_dim_e = nn.Sequential(
            nn.Conv1d(self.embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.increase_dim_d = nn.Sequential(
            nn.Conv1d(self.embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        # Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(dim=self.embed_dim, num_heads=6, mlp_ratio=2., qkv_bias=None, qk_scale=0.,drop=0., attn_drop=0.)
            for i in range(self.depth[1])])

        # Reconstruction
        if self.num_points is not None:
            self.factor = self.num_points // self.num_query
            assert self.num_points % self.num_query == 0
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point
        else:
            self.factor = self.fold_step**2
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)

        self.increase_dim_2 = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, xyz_label):

        # FAPE
        coor, f, fps_idx = self.FAPE(xyz_label.transpose(1, 2).contiguous())

        # Label-embeding
        pos = self.pos_embed(coor).transpose(1, 2)  # label embed
        x = self.input_proj(f).transpose(1, 2)

        # Geometry-aware transformer encoder
        knn_index = get_knn_index(coor)
        for i, blk in enumerate(self.encoder):
            if i < self.knn_layer:
                x = blk(x + pos, knn_index)  # B N C
            else:
                x = blk(x + pos)

        # Point morphing
        local_features = self.increase_dim_e(x.transpose(1, 2))  # B 1024 N
        global_feature = torch.max(local_features, dim=-1)[0]  # B 1024
        global_feature = global_feature.unsqueeze(1).repeat(1, self.num_query, 1)  # B, 256, 1024
        combined_features = torch.cat([local_features.transpose(1, 2), global_feature], dim=2)
        coarse_relative = self.coarse_pred(combined_features)  # B M C(3)
        coarse_point_cloud = coarse_relative + coor[:, :3, :].transpose(1, 2)

        # Geometry-aware transformer decoder
        new_knn_index = get_knn_index(coarse_point_cloud.transpose(1, 2).contiguous())
        query_feature = torch.cat([
            # global_feature.unsqueeze(1).repeat(1, self.num_query, 1),
            global_feature,
            coarse_point_cloud], dim=-1)  # B M C+3
        q = self.mlp_query(query_feature.transpose(1, 2)).transpose(1, 2)  # B M C

        for i, blk in enumerate(self.decoder):
            if i < self.knn_layer:
                q = blk(q, x, new_knn_index, new_knn_index)  # B M C
            else:
                q = blk(q, x)

        # Patch reconstruction
        B, M, C = q.shape
        global_feature = self.increase_dim_d(q.transpose(1, 2)).transpose(1, 2)  # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        rebuild_feature = self.reduce_map(rebuild_feature)  # B M C
        relative_xyz = self.decode_head(rebuild_feature)  # B M S 3
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3
        pred_fine = rebuild_points.reshape(B, -1, 3).contiguous()

        pred_coarse = coarse_point_cloud.contiguous()
        ret = (pred_coarse, pred_fine, coor[:, :3, :].transpose(1, 2), fps_idx,coor[:, 3:, :].transpose(1, 2))
        return ret

class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()
        self.loss_func = ChamferDistanceL1()

    def forward(self, ret, gt):

        loss_fine = self.loss_func(ret[1], gt[0])
        loss_coarse = torch.mean(torch.sqrt(torch.sum((ret[0] - gt[1]) ** 2, axis=2)))

        return loss_coarse, loss_fine

"""
#=============================================
Model Config
#=============================================
"""
def parse_arg(argv=None):
    parser = argparse.ArgumentParser('FracFormer')

    # Embed
    parser.add_argument('--num_pred', default=4096, type=int, help='num of center points')
    parser.add_argument('--num_query', default=256, type=int, help='num of group size')
    parser.add_argument('--knn_layer', default=1, type=int, help='dim of group feat')
    parser.add_argument('--trans_dim', default=384, type=int, help='dim of group feat')

    parser.add_argument('--embed_dim', default=768, type=int, help='dim of group feat')
    parser.add_argument('--One_hot_dim', default=7, type=int, help='dim of group feat')
    # Decoder
    args = parser.parse_args(argv)
    return args
