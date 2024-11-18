# from pathlib import Path
# from types import SimpleNamespace

# import cvxpy as cp
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import yaml
# from einops import rearrange, repeat
# from IPython.display import display
# from pandas import DataFrame, set_option
# from scipy.linalg import matmul_toeplitz

# set_option('display.max_columns', None)
# set_option('display.max_rows', None)

# data_dir = Path.cwd().parents[1].joinpath('Datasets', 'DYAN')


# def find_alpha(A, Y, gamma, tau):
#     A = rearrange(A, 'b h w -> (b h) w')
#     Y = rearrange(Y, 'h w -> (h w)')
#     coh = np.absolute(np.dot(A.T, Y))
#     alpha0 = np.amax(coh) / tau  # value for which the solution is zero
#     alpha = alpha0 / gamma
#     return alpha / 2


# def loss(y, AC, alpha):
#     return alpha * cp.norm2(y-AC)**2

# def regularization(c, lambda1):
#     lambda2 = (1-lambda1) / 2
#     return lambda1 * cp.norm1(c) + lambda2 * cp.norm2(c)**2

# def solve_lasso(Yn, An, lambda1, Np, T, alpha, m, threshold):
#     C_pred = []
#     if Yn.ndim < 3:
#         Yn = rearrange(Yn, 'h w -> h w 1')
#     for i in range(0,m):
#         y_i = Yn[i]
#         # y_i = Yn[T*(i):T*(i+1)]
#         reg = 0
#         AC_i = 0
#         for j in range(0,m):
#             c = cp.Variable([Np, 1])
#             C_pred.append(c)
#             reg = reg + regularization(c, lambda1)
#             AC_i = AC_i + (An[i][:,Np*j:Np*j+Np] @ c)
            
#         obj = loss(y_i, AC_i, alpha) + reg

#         problem = cp.Problem(cp.Minimize(obj))
#         problem.solve(solver=cp.MOSEK)

#     ans = [arr.value for arr in C_pred]
    
#     ans = np.array(ans).squeeze()
#     ans = np.where(ans > threshold, ans, 0)
#     return ans

# def nice_display(arr):
#     if arr.ndim <= 2:
#         display(DataFrame(arr))
#     else:
#         display(DataFrame(np.squeeze(arr)))
        


# def normalize_matrix(arr):
#     column_norm = np.linalg.norm(arr, ord=2.0, axis=0, keepdims=True)
#     An = arr / column_norm
#     return An

# def generate_sparse_vector(m, n_poles, Np, as_ones=False):
#     """ Generates a set of sparse vectors

#     Args:
#         m (int): number of points
#         n_poles (int): number of atoms for each trajectory
#         Np (int): width of the dictionary
#         as_ones (bool, optional): chooses to use unity for poles or generate a random number. Defaults to False.

#     Returns:
#         array: an vector of coefficients of shape (m**2, Np)
#     """
#     m2 = np.power(m, 2)
#     if as_ones is True:
#         c_gt = np.ones([m2, n_poles])
#     else:
#         c_gt = np.random.uniform(-1,1,[m2, n_poles])
#     atoms_gt = np.random.randint(0, Np, [m2, n_poles])
#     atoms_gt.sort()
#     C = np.zeros([m2, Np])
#     for i in range(m2):
#         for j in range(n_poles):
#             C[i,atoms_gt[i,j]] = c_gt[i,j]
#     # print(C[C.nonzero()])
#     return C
    
# def self_interactions_test(C, m):
#     """ sets interactions to only self loops (diagonal) and each point only effecting the first point

#     Args:
#         C (array): array of shape (m**2, Np)
#         m (int): number of points

#     Returns:
#         array: a sparse vector of shape (m**2, Np)
#     """
#     C_block = rearrange(C, '(b h) w -> b h w', b=m)
#     C_ = np.zeros(C_block.shape)
#     C_[0,:] = C_block[0,:]
#     for i in range(m):
#         C_[i,i] = C_block[i,i]
        
#     C_ = rearrange(C_, 'b h w -> (b h) w')  
#     # print(C_.shape)
#     return C_

# def toeplitz_upper_zero(y, D, T):
#     r = np.pad(y[0], (0,T-1), 'constant', constant_values=0) # make the upper triangle zeros
#     return matmul_toeplitz((y, r), D, check_finite=False)

# def generate_A(y, m, D, T):
#     """Generates Matrix A

#     Args:
#         y (array): 4 x 10 x 1 
#         m (int): number of points
#         D (array): dyan dictionary
#         T (int): number of frames

#     Returns:
#         list: a list of m elements; each row of matrix A (T, m*Np)
#     """
#     y_block = rearrange(y, '(b h) w c -> b h w c', b=m)
#     toeplitz_mats = [toeplitz_upper_zero(y_block[i,i], D, T) for i in range(m)]
#     A_blocks = [toeplitz_mats[:i] + [D] + toeplitz_mats[i+1:] for i in range(m)]
#     A_rows = [np.concatenate(sublist, axis=1) for sublist in A_blocks]
#     # print(A_rows[0].shape)
#     return A_rows, A_blocks

# def difference(a, b, tol):
#     return (np.absolute(a-b < tol)).all()

# def generate_test_data(m, n_poles, Np, D, T, unity_poles=True):
#     C = generate_sparse_vector(m, n_poles, Np, as_ones=unity_poles)
#     C_gt = self_interactions_test(C, m)
#     y = np.matmul(D, rearrange(C_gt, 'h w -> h w 1')) # shape=(m**2, T, 1)
#     A_rows, _ = generate_A(y, m, D, T)
#     # print(np.array(A_rows).shape)
#     # print(rearrange(C_gt, '(b h) w -> b (h w) 1', b=m).shape)
#     Y = np.matmul(np.array(A_rows), rearrange(C_gt, '(b h) w -> b (h w) 1', b=m)).squeeze() # shape=(4,10)

#     A = np.zeros([m*T, m*m*Np])
#     for i, row in enumerate(A_rows):
#         indx = m*Np
#         A[i*T:(i+1)*T, i*indx:(i+1)*indx] = row
    
    
#     return Y, np.array(A_rows), C_gt, A

# class YamlParser():
#     def __init__(self, path=None) -> None:
#         self.path=path
#         pass
        
#     def dict_to_namespace(self, d):
#         if isinstance(d, dict):
#             for k, v in d.items():
#                 d[k] = self.dict_to_namespace(v)
#             return SimpleNamespace(**d)
#         elif isinstance(d, list):
#             return [self.dict_to_namespace(i) for i in d]
#         else:
#             return d

#     def load_config(self):
#         with open(self.path, 'r') as f:
#             self.config_file = yaml.safe_load(f)

#     def unpack_config(self):
#         # Check if the top-level structure is a dictionary with multiple keys or not
#         if isinstance(self.config_file, dict) and len(self.config_file) > 1:
#             # Multiple top-level keys, use logic similar to unpack_config_toplevel
#             top_level_namespaces = []
#             names = []
#             for key, value in self.config_file.items():
#                 # Convert each top-level dictionary into a SimpleNamespace
#                 namespace = self.dict_to_namespace(value)
#                 setattr(self, key, namespace)  # Set each namespace as an attribute for easy access
#                 names.append(key)
#                 top_level_namespaces.append(namespace)
#             # Return the list of top-level namespaces and their names
#             return [names, top_level_namespaces]
#         else:
#             # Single top-level structure, use logic similar to unpack_config
#             SimpleNamespace(**{k: self.dict_to_namespace(v) for k, v in self.config_file.items()})  
#             # return [self.config_file.keys(), self.dict_to_namespace(self.config_file)]
#             return [self.config_file.keys(), SimpleNamespace(**{k: self.dict_to_namespace(v) for k, v in self.config_file.items()})]

#     def list_attributes(self, value=None):
#         if value is None:
#             attributes = vars(self.names)
#         else:
#             attributes = vars(value)
#         for key, value in attributes.items():
#             print(f"Attribute: {key}")
#             if isinstance(value, SimpleNamespace):
#                 print(f"Nested attributes under {key}:")
#                 self.list_attributes(value)
                
#     def parse(self,path=None):
#         if path is not None:
#             self.path = path
        
#         self.load_config()
#         self.names = self.unpack_config()
#         return self.names

# class Dyan():
#     def __init__(self, path, config) -> None:
#         self.path = path
#         self.filename = config.filename
#         self.T = config.T
#         self.Np = config.Np
#         self.load_raw_data()
#         self.generate_dictionary()
        
        
#     def load_raw_data(self):
#         self.raw_data = np.load(self.path.joinpath(self.filename))
#         self.r = self.raw_data[0,:]
#         self.theta = self.raw_data[1,:]
        
        
#     def generate_dictionary(self):
#         self.dictionary = np.zeros([self.T,self.Np])
        
#         for i in range(0,self.T):
#             r_pow = np.power(self.r,i)
#             self.dictionary[i,:] = np.concatenate([[1], r_pow * np.cos(i*self.theta),
#                                      r_pow * np.sin(i*self.theta)])
            
    