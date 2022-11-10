import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_w(x):
    dim = x.size(1) // 4
    return torch.split(x, dim, dim=1)[0]

def get_x(x):
    dim = x.size(1) // 4
    return torch.split(x, dim, dim=1)[1]

def get_y(x):
    dim = x.size(1) // 4
    return torch.split(x, dim, dim=1)[2]

def get_z(x):
    dim = x.size(1) // 4
    return torch.split(x, dim, dim=1)[3]

def quat_conjugate(x):
    dim = x.size(-1) // 4
    w, x, y, z = torch.split(x, dim, dim=-1)
    return torch.cat([w, -x, -y, -z], dim=-1)

def complex_conjugate(x):
    dim = x.size(-1) // 2
    a, b = torch.split(x, dim, dim=-1)
    return torch.cat([a, -b], dim=-1)

def Hamilton_conjugate(x):
    dim = x.size(-1) // 4
    w, x, y, z = torch.split(x, dim, dim=-1)
    return torch.cat([w, -(x), -(y), -(z)], dim=-1)

def Hermitian_conjugate(x):
    dim = x.size(-1) // 4
    w, x, y, z = torch.split(x, dim, dim=-1)
    return torch.cat([complex_conjugate(w), -complex_conjugate(x), -complex_conjugate(y), -complex_conjugate(z)], dim=-1)

def quat_mul(a, b):
    assert a.size(-1) == b.size(-1)
    dim = a.size(-1) // 4
    s_a, x_a, y_a, z_a = torch.split(a, dim, dim=-1)
    s_b, x_b, y_b, z_b = torch.split(b, dim, dim=-1)

    A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b  # 0, 1, 2, 3
    B = s_a * x_b + x_a * s_b + y_a * z_b - z_a * y_b  # 1, 0, 3, 2
    C = s_a * y_b - x_a * z_b + y_a * s_b + z_a * x_b  # 2, 3, 0, 1
    D = s_a * z_b + x_a * y_b - y_a * x_b + z_a * s_b  # 3, 2, 1, 0

    return torch.cat([A,B,C,D], dim=-1)

def complex_mul(a, b):
    assert a.size(-1) == b.size(-1)
    dim = a.size(-1) // 2
    a_1, a_2 = torch.split(a, dim, dim=-1)
    b_1, b_2 = torch.split(b, dim, dim=-1)

    A = a_1 * b_1 - a_2 * b_2
    B = a_1 * b_2 + a_2 * b_1 

    return torch.cat([A,B], dim=-1)

def biquaternion_mul(w_a, x_a, y_a, z_a, w_b, x_b, y_b, z_b):
    A = complex_mul(w_a,w_b) - complex_mul(x_a,x_b) - complex_mul(y_a,y_b) - complex_mul(z_a,z_b)  
    B = complex_mul(w_a,x_b) + complex_mul(x_a,w_b) + complex_mul(y_a,z_b) - complex_mul(z_a,y_b)  
    C = complex_mul(w_a,y_b) - complex_mul(x_a,z_b) + complex_mul(y_a,w_b) + complex_mul(z_a,x_b)  
    D = complex_mul(w_a,z_b) + complex_mul(x_a,y_b) - complex_mul(y_a,x_b) + complex_mul(z_a,w_b)  
    return torch.cat([A, B, C, D], dim=-1)

def get_norm(x, y, nums):
    dim1 = x.size(-1) // nums
    x_split = torch.split(x, dim1, dim=-1)
    dim2 = y.size(-1) // nums
    y_split = torch.split(y, dim2, dim=-1)
    norm1, norm2 = 0, 0
    for i in x_split:
        norm1 += i**2
    for j in y_split:
        norm2 += j**2
    return torch.sqrt(norm1 + norm2)