
import os
from torch.utils.data import Dataset
import numpy as np
import copy
from scipy.spatial.transform import Rotation

def list_file_names(directory):#返回包含目录下所有文件完整路径的列表
    file_names = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return file_names

def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype(np.int32)

def pc_normalize_params(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc,centroid,m

def randomMatrix(points,max_angle=30,max_tran=30,type='normal'):
    #生成一个 ​​刚体变换矩阵​​（4x4齐次坐标矩阵），包含随机旋转和平移，且变换中心为点云的几何中心。
    if type =='uniform':
        #生成三个随机旋转角（欧拉角）和平移向量
        random_angles = np.random.uniform(-max_angle, max_angle, size=(3))
        random_translations = np.random.uniform(-max_tran, max_tran, size=(3)) # 正态分布
    else:
        random_angles = np.random.normal(loc=0, scale=max_angle, size=(3))  # 正态分布
        random_translations = np.random.normal(loc=0, scale=max_tran, size=(3))  # 正态分布
    random_rotations = Rotation.from_euler('xyz', random_angles, degrees=True).as_matrix()#将随机生成的欧拉角转换为 3x3 旋转矩阵。
    #若 random_angles = [30, 15, -10]，则依次绕 X 轴转 30 度，Y 轴转 15 度，Z 轴转 -10 度。

    Trandom = np.eye(4)
    Trandom[:3, :3] = random_rotations
    Trandom[:3, 3] = random_translations

    center = np.mean(points, axis=0)#将点云中心平移到原点，确保变换围绕中心进行。
    Tcenter = np.identity(4)
    Tcenter[:3, 3]= -center#构建平移矩阵 Tcenter​​，当应用此矩阵时，点云会被平移至 ​​坐标系原点​​（质心对齐原点）。

    
    Trandom = np.dot(np.linalg.inv(Tcenter),np.dot(Trandom,Tcenter))#将初始变换矩阵 Trandom 修正为以点云中心为参考系的变换。
    return Trandom

def apply_transform(deformShape,Trand):

    N = deformShape.shape[0]
    deformShape_homogeneous = np.hstack((deformShape, np.ones((N, 1))))

    deformShape_transformed = np.dot(Trand, deformShape_homogeneous.T).T

    deformShape_transformed = deformShape_transformed[:, :3]

    return deformShape_transformed

def FracturePose(Target,Labelsample):

    deformShape = copy.deepcopy(Target)

    FracNum = np.max(Labelsample)
    Tramdom = np.identity(4)

    for Fracidx in range(FracNum):
        indices = Labelsample == Fracidx+1
        Tramdom = randomMatrix(deformShape[indices],max_angle=30,max_tran=15,type='uniform')
        deformShape[indices]=apply_transform(deformShape[indices],Tramdom)

    return deformShape,Tramdom

def preprocess(Fratures,Target,Labelsample,npoints):

    sampleIDX = np.random.choice(Target.shape[0], size=10000, replace=True)
    Fratures = Fratures[sampleIDX, :]
    Labelsample = Labelsample[sampleIDX]
    Target = Target[sampleIDX, :]

    perm = np.random.permutation(len(Labelsample))
    Fratures = Fratures[perm, :]
    Labelsample = Labelsample[perm]
    Target = Target[perm, :]

    sampleIDX2 = farthest_point_sample(Fratures,npoints)
    Fratures = Fratures[sampleIDX2,:]
    Labelsample = Labelsample[sampleIDX2]

    Fratures, center, scale = pc_normalize_params(Fratures)
    Target = (Target - center)/ scale

    return Fratures,Target,Labelsample,sampleIDX2

def random_rotation_matrix():
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def randSheer(scale_range=0,shear_range=0):
    #模拟骨骼碎片在受力时的局部尺寸变化（如压缩或拉伸导致的长短、粗细改变）。例如，骨折端可能因撞击产生微小的缩短（缩放因子 <1）或错位拉伸（缩放因子> 1）
    scale_x = 1.0 + np.random.uniform(-scale_range, scale_range)
    scale_y = 1.0 + np.random.uniform(-scale_range, scale_range)
    scale_z = 1.0 + np.random.uniform(-scale_range, scale_range)#模拟骨骼碎片在 x/y/z 轴上的局部拉伸或压缩（如骨折端挤压导致的尺寸变化）。
    #模拟骨骼碎片在受到剪切力（如扭转、侧方撞击）时的非均匀错位。例如，踝关节骨折中，腓骨碎片可能因扭转力发生水平方向的剪切错位（对应矩阵中的shear_x或shear_y）
    shear_x = np.random.uniform(-shear_range, shear_range)#模拟骨折面的错位变形（如碎片沿某一平面滑动导致的倾斜）。
    shear_y = np.random.uniform(-shear_range, shear_range)
    shear_z = np.random.uniform(-shear_range, shear_range)
    transformation_matrix = np.array([[scale_x, shear_x, 0],## X轴方向：缩放+X-Y平面剪切
                                      [shear_y, scale_y, 0],
                                      [shear_z, shear_z, scale_z]])
    return transformation_matrix

def global_deform(points, scale_range=0.1, shear_range=0.2):
    
    center = np.mean(points,axis=0)
    centered_points = points - center#将点云重心平移至原点

    rotation_matrix = random_rotation_matrix()
    rotated_points = centered_points @ rotation_matrix.T

    shear_matrix = randSheer(scale_range, shear_range)
    transformed_points = rotated_points @ shear_matrix.T
    
    #计算旋转矩阵的逆矩阵，将剪切变换后的点云应用逆旋转，也就是将点云旋转回原来的方向，但已经应用了剪切和缩放变换。
    #这样做的目的是为了保持变换后的点云在原来的方向上，但形状已经被修改。然后，将点云平移回原来的中心位置。
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    transformed_points = transformed_points @ inverse_rotation_matrix.T

    final_points = transformed_points + center

    return final_points

def add_jitter_all(points, jitter_level=0.5, probability=0.3):

    num_points = points.shape[0]
    jitter = np.random.randn(num_points, 3) * jitter_level
    mask = np.random.rand(num_points) < probability
    points[mask] += jitter[mask]
    return points

def sort_labels_by_frequency(label):

    unique_labels, counts = np.unique(label, return_counts=True)

    sorted_indices = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_indices]

    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}

    sorted_label_array = np.vectorize(label_mapping.get)(label)

    return sorted_label_array,len(unique_labels)

class DataLoader(Dataset):
    def __init__(self, type):
        self.npoints = 2048
        self.BoneType = type
        self.dataPath = os.path.join('data',self.BoneType)
        # functions
        self.Tinit = True
        self.Elastic = True
        self.jitter = True
        self.type = type

        # output list
        self.frac_list = list_file_names(self.dataPath)

        print(len(self.frac_list))
        self.Datalen = len(self.frac_list)

    def __len__(self):
        return self.Datalen

    def _get_item(self,idx):

        Fractures = np.loadtxt(self.frac_list[idx],delimiter=',')#读取.txt文件
        DeformShape,Labelsample = Fractures[:,:3],Fractures[:,3].astype(np.int64)#分别获取所有行的前三列（索引0,1,2），表示三维点云坐标。所有行的第四列（索引3），转换为整数类型，表示每个点的分类标签。
        if self.Elastic:#对点云施加 ​​非刚性形变​​，模拟物体弯曲、拉伸等弹性变化
            DeformShape = global_deform(DeformShape, scale_range=0.1, shear_range=0.1)
        if self.Tinit:
            Trandom = randomMatrix(DeformShape, max_angle=30, max_tran=30)
            DeformShape = apply_transform(DeformShape, Trandom)
        if self.jitter:
            DeformShape = add_jitter_all(DeformShape, probability=0.1)

        Target = copy.deepcopy(DeformShape)

        Fratures, Tfrac = FracturePose(DeformShape, Labelsample)

        Fratures, Target, Labelsample,fps_0 = preprocess(Fratures, Target, Labelsample, self.npoints)
        Labelsample,Label_len = sort_labels_by_frequency(Labelsample)

        data_dict = {
            'idx': idx,
            'Fractures': Fratures.astype(np.float32),
            'Target': Target.astype(np.float32),
            'Label': Labelsample.astype(np.int64),
            'fps_0': fps_0.astype(np.int32),
        }

        return data_dict

    def __getitem__(self, index):
        return self._get_item(index)


