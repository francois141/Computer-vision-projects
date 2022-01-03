import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  for i in range(num_corrs):
    # Fill first row
    constraint_matrix[2*i,0] = 0
    constraint_matrix[2*i,1] = 0
    constraint_matrix[2*i,2] = 0
    constraint_matrix[2*i,3] = 0
    constraint_matrix[2*i,4] = -points3D[i,0]
    constraint_matrix[2*i,5] = -points3D[i,1]
    constraint_matrix[2*i,6] = -points3D[i,2]
    constraint_matrix[2*i,7] = -points3D[i,3]
    constraint_matrix[2*i,8] = points2D[i,1]*points3D[i,0]
    constraint_matrix[2*i,9] = points2D[i,1]*points3D[i,1]
    constraint_matrix[2*i,10] = points2D[i,1]*points3D[i,2]
    constraint_matrix[2*i,11] = points2D[i,1]*points3D[i,3]
    # Fill second row
    constraint_matrix[2*i+1,0] = points3D[i,0]
    constraint_matrix[2*i+1,1] = points3D[i,1]
    constraint_matrix[2*i+1,2] = points3D[i,2]
    constraint_matrix[2*i+1,3] = 1
    constraint_matrix[2*i+1,4] = 0
    constraint_matrix[2*i+1,5] = 0
    constraint_matrix[2*i+1,6] = 0
    constraint_matrix[2*i+1,7] = 0
    constraint_matrix[2*i+1,8] = -points2D[i,0]*points3D[i,0]
    constraint_matrix[2*i+1,9] = -points2D[i,0]*points3D[i,1]
    constraint_matrix[2*i+1,10] = -points2D[i,0]*points3D[i,2]
    constraint_matrix[2*i+1,11] = -points2D[i,0]*points3D[i,3]
    
  return constraint_matrix