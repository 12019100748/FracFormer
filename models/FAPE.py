import torch
from torch import nn
from pointnet2_ops import pointnet2_utils

from knn_cuda import KNN
knn = KNN(k=16, transpose_mode=False)

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

def fps_label(pc_label, feat, num):
    # calculate label ratio
    encoded_label = pc_label[:, :, 3:]
    label_counts = encoded_label.sum(dim=1)
    label_ratio = label_counts.float() / label_counts.sum(dim=1, keepdim=True)
    initial_sample_counts = torch.ceil(label_ratio * num).int()

    difference = initial_sample_counts.sum(dim=1) - num
    reduction = torch.min(difference, initial_sample_counts[:, 0] - 1)
    initial_sample_counts[:, 0] -= reduction

    # fps
    batch_size, num_points, _ = pc_label.shape
    #all_fps_points = []
    all_fps_indices = []
    pc_label = torch.cat((pc_label,feat.permute(0,2,1)), 2)
    num_classes = encoded_label.shape[2]

    for i in range(batch_size):
        fps_indices_per_batch = []
        for label in range(num_classes):
            mask = encoded_label[i, :, label].bool()
            current_pc_label = pc_label[i, mask]

            num_samples = initial_sample_counts[i, label].item()
            if num_samples > 0 and current_pc_label.shape[0] > 0:
                fps_idx = pointnet2_utils.furthest_point_sample(current_pc_label[:,:3].unsqueeze(0).contiguous(), num_samples)
                extracted_indices = torch.where(mask)[0][fps_idx]
                fps_indices_per_batch.append(extracted_indices.squeeze(0))

        if fps_indices_per_batch:
            all_fps_indices.append(torch.cat(fps_indices_per_batch, dim=0))

    #final_sampled_pc = torch.stack(all_fps_points)
    final_fps_indices = torch.stack(all_fps_indices).to(torch.int32)
    final_sampled_pc = pointnet2_utils.gather_operation(pc_label.transpose(1, 2).contiguous(),
                                                        final_fps_indices).transpose(1, 2).contiguous()
    return final_sampled_pc.transpose(1, 2).contiguous(),final_fps_indices

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
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

class DGCNN_FAPE(nn.Module):
    def __init__(self, num_center):
        self.num_center = num_center
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(10, 10, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(20, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )
        self.num_features = 128
    @staticmethod
    def fps_downsample_label(coor, x, num_group):
        B,C,N=coor.shape
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3

        new_combined_x,fps_idx = fps_label(xyz,x, num_group)

        new_coor = new_combined_x[:, :C]
        new_x = new_combined_x[:, C:]

        return new_coor, new_x ,fps_idx

    @staticmethod
    def get_label_neibour(coor_q, coor_k):
        k = 16
        batch_size, num_dims, num_points_q = coor_q.shape
        _, _, num_points_k = coor_k.shape

        labels_q = coor_q[:, 3:, :]  # [B, 5, npoints_q]
        labels_k = coor_k[:, 3:, :]  # [B, 5, npoints_k]

        all_neighbors_indices = torch.full((batch_size, num_points_q, k), -1, dtype=torch.int64, device=coor_q.device)

        for b in range(batch_size):

            active_labels = torch.any(labels_q[b].bool(), dim=1)
            # save_to_txt(coor_k[b, :3, :].transpose(0, 1), f'batch_{b}_coor_k.txt')

            for label in torch.where(active_labels)[0]:
                mask_q = labels_q[b, label, :].bool()
                mask_k = labels_k[b, label, :].bool()

                coor_q_label = coor_q[b, :3, mask_q]
                #x_q_label = x_q[b, :, mask_q]
                coor_k_label = coor_k[b, :3, mask_k]
                #x_k_label = x_k[b, :, mask_k]

                if mask_k.sum() >= k:
                    _, idx_knn = knn(coor_k_label.unsqueeze(0), coor_q_label.unsqueeze(0))
                    idx_knn = idx_knn.squeeze(0)
                else:
                    repeat_times = (k // coor_k_label.size(1)) + 1
                    idx_knn = torch.arange(coor_k_label.size(1), device=coor_k_label.device).repeat(repeat_times)[:k]
                    idx_knn = idx_knn.unsqueeze(1).repeat(1, coor_q_label.size(1))

                extracted_indices = torch.where(mask_k)[0][idx_knn]
                all_neighbors_indices[b, mask_q, :] = extracted_indices.permute(1, 0)

        return all_neighbors_indices  # Adjust dimensions to [B, np, k, 2C]

    @staticmethod
    def get_graph_feature_label(coor_q, x_q, coor_k, x_k):
        k = 16
        batch_size, num_dims, num_points_q = x_q.shape
        _, _, num_points_k = x_k.shape

        labels_q = coor_q[:, 3:, :]  # [B, 5, npoints_q]
        labels_k = coor_k[:, 3:, :]  # [B, 5, npoints_k]

        all_features = torch.zeros(batch_size, num_dims * 2, num_points_q, k, device=x_q.device)
        all_neighbors_indices = torch.full((batch_size, num_points_q, k), -1, dtype=torch.int64, device=x_q.device)

        for b in range(batch_size):
            active_labels = torch.any(labels_q[b].bool(), dim=1)

            for label in torch.where(active_labels)[0]:
                mask_q = labels_q[b, label, :].bool()
                mask_k = labels_k[b, label, :].bool()

                coor_q_label = coor_q[b, :3, mask_q]
                x_q_label = x_q[b, :, mask_q]
                coor_k_label = coor_k[b, :3, mask_k]
                x_k_label = x_k[b, :, mask_k]

                if mask_k.sum() >= k:
                    _, idx_knn = knn(coor_k_label.unsqueeze(0), coor_q_label.unsqueeze(0))
                    idx_knn = idx_knn.squeeze(0)
                else:
                    repeat_times = (k // coor_k_label.size(1)) + 1
                    idx_knn = torch.arange(coor_k_label.size(1), device=x_q.device).repeat(repeat_times)[:k]
                    idx_knn = idx_knn.unsqueeze(1).repeat(1, coor_q_label.size(1))

                all_neighbors_indices[b, mask_q, :] = idx_knn.permute(1, 0)

                gathered_x_k = x_k_label[:, idx_knn].squeeze(0)  # [C, k]
                x_diff = gathered_x_k.transpose(1,2) - x_q_label.unsqueeze(2)  # [C, num_q_points, k]

                combined_feature = torch.cat([x_diff, x_q_label.unsqueeze(2).expand_as(x_diff)],
                                             dim=0)  # [2C, num_q_points, k]
                all_features[b, :, mask_q, :] = combined_feature  # Reshape to match dimensions


        return all_features,all_neighbors_indices # Adjust dimensions to [B, np, k, 2C]

    def forward(self, x):
        # x: bs, 3, np

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x
        f = self.input_trans(x)

        f,_ = self.get_graph_feature_label(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q,fps_idx_1 = self.fps_downsample_label(coor, f, self.num_center[0])
        f,_ = self.get_graph_feature_label(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f,_ = self.get_graph_feature_label(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q,fps_idx_2 = self.fps_downsample_label(coor, f, self.num_center[1])  # paper recommand 256
        f,nei_idx = self.get_graph_feature_label(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        return coor, f,(fps_idx_1,fps_idx_2)
