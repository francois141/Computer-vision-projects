import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # TODO
  # Normalize coordinates (to points on the normalized image plane)
  K_inv = np.linalg.inv(K)

  # These are the keypoints on the normalized image plane (not to be confused with the normalization in the calibration exercise)
  normalized_kps1 = (K_inv @ MakeHomogeneous(im1.kps,ax=1).transpose()).transpose()
  normalized_kps2 = (K_inv @ MakeHomogeneous(im2.kps,ax=1).transpose()).transpose()

  # TODO
  # Assemble constraint matrix
  constraint_matrix = np.zeros((matches.shape[0], 9))

  for i in range(matches.shape[0]):
    index1 = matches[i,0]
    index2 = matches[i,1]
    constraint_matrix[i,0] = normalized_kps1[index1,0] * normalized_kps2[index2,0]
    constraint_matrix[i,1] = normalized_kps1[index1,0] * normalized_kps2[index2,1]
    constraint_matrix[i,2] = normalized_kps1[index1,0] 
    constraint_matrix[i,3] = normalized_kps1[index1,1] * normalized_kps2[index2,0]
    constraint_matrix[i,4] = normalized_kps1[index1,1] * normalized_kps2[index2,1]
    constraint_matrix[i,5] = normalized_kps1[index1,1]
    constraint_matrix[i,6] = normalized_kps2[index2,0]
    constraint_matrix[i,7] = normalized_kps2[index2,1]
    constraint_matrix[i,8] = 1
  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  # TODO
  # Reshape the vectorized matrix to it's proper shape again
  E_hat = np.zeros(shape=(3,3))
  E_hat[0] = vectorized_E_hat[0:3]
  E_hat[1] = vectorized_E_hat[3:6]
  E_hat[2] = vectorized_E_hat[6:9]

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily

  U,S,V = np.linalg.svd(E_hat)
  S[2] = 0
  E = U @ np.diag(S) @ V

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]

    assert(abs(kp1.transpose() @ E @ kp2) < 0.01)

  return E

  
def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`
  # P1 @ MakeHomegenous(points3D[i])[2]<0 and P2 @ MakeHomegenous(points3D[i])[2]<0

  pts_behind = [i for i in range(len(points3D)) if (P1 @ MakeHomogeneous(points3D[i]))[2]<0 or (P2 @ MakeHomogeneous(points3D[i]))[2]<0]
  new_points3D = np.delete(points3D, pts_behind, 0)
  new_im1_corrs = np.delete(im1_corrs, pts_behind, 0)
  new_im2_corrs = np.delete(im2_corrs, pts_behind, 0)

  return new_points3D, new_im1_corrs, new_im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  normalized_points2D = np.linalg.inv(K) @ MakeHomogeneous(points2D,ax=1).transpose()
  normalized_points2D = normalized_points2D.transpose()
  points3D = MakeHomogeneous(points3D,ax=1)
  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  image = images[image_name]
  points3D = np.zeros((0,3))
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}

  for reg_img in registered_images:

    e_matches = GetPairMatches(image_name, reg_img, matches)
    new_points3D, im1_corrs, im2_corrs = TriangulatePoints(K, image, images[reg_img], e_matches)
    if image_name not in corrs:
      corrs[image_name] = im1_corrs
    else:
      corrs[image_name] = np.append(corrs[image_name], im1_corrs)
    corrs[reg_img] = im2_corrs
    points3D = np.append(points3D, new_points3D, 0)

  return points3D, corrs