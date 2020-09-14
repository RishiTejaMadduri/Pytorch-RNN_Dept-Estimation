#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2 as cv


# In[2]:


def save_color_depth(depth,filename):
    plt.imsave(filename, depth, cmap= 'plasma')


# In[3]:


def gray2rgb(im, cmap='gray'):
    cmap=plt.get_cmap(cmap)
    rgba_img=cmap(im.astype(np.float32))
    rgb_img=np.delete(rgba_img,3,2)
    return rgb_img


# In[4]:


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalize=None, cmap='gray'):
    
    #Convert to disparity
    depth=1./(depth+1e-6)
    
    if normalizer is not None:
        depth=depth/normalizer
    else:
        depth=depth/(np.percentile(depth, pc)+1e-6)
        
    depth = np.clip(depth,0,1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0]*(1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth


# In[8]:


def image_gradient_direction(img):
    eps=1e-11
    sobelx=cv.Sobel(img, cv.CV_64F,1,1,ksize=5)+eps
    sobely=cv.Sobel(img, cv.CV_64F,0,1,ksize=5)+eps
    
    direction=np.arctan(sobely/sobelx)*180/np.pi
    direction[(direction<=0) & (sobelx<0)]+=180
    direction[(direction<0) & (sobely<0)]+=360
    direction[(direction>0) & (sobelx<0)]+=180
    
    return direction


# In[3]:


#Original-Rewritten
def euler2mat(z,y,x):
    """Converts euler angles to rotation matrix
       TODO: remove the dimension for 'N' (deprecated for converting all source
             poses altogether)
       Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
      Args:
          z: rotation angle along z axis (in radians) -- size = [B, N]
          y: rotation angle along y axis (in radians) -- size = [B, N]
          x: rotation angle along x axis (in radians) -- size = [B, N]
      Returns:
          Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
      """
    
    B = z.size(0)
    N = 1
    z = torch.unsqueeze(torch.unsqueeze(z,-1), -1)
    y = torch.unsqueeze(torch.unsqueeze(y,-1), -1)
    x = torch.unsqueeze(torch.unsqueeze(x,-1), -1)
    
    zeros = torch.zeros((B,N,1,1))
    ones = torch.ones((B,N,1,1))
    
    cosz = torch.cos(z)
    sinz = torch.sin(z)
    
    rotz_1 = torch.cat([cosz, -sinz, zeros], axis=3)
    rotz_2 = torch.cat([cosz, -sinz, zeros], axis=3)
    rotz_3 = torch.cat([cosz, -sinz, zeros], axis=3)
    zmat =  torch.cat([rotz_1, rotz_2, rotz_3], axis=2)
    
    cosy = torch.cos(y)
    siny = torch.sin(y)
    
    roty_1 = torch.cat([cosy, zeros, siny], axis=3)
    roty_2 = torch.cat([zeros, ones, zeros], axis=3)
    roty_3 = torch.cat([-siny,zeros, cosy], axis=3)
    ymat = torch.cat([roty_1, roty_2, roty_3], axis=2)
    
    cosx = torch.cos(x)
    sinx = torch.sin(x)
    
    rotx_1 = torch.cat([ones, zeros, zeros], axis=3)
    rotx_2 = torch.cat([zeros, cosx, -sinx], axis=3)
    rotx_3 = torch.cat([zeros, sinx, cosx], axis=3)
    xmat = torch.cat([rotx_1, rotx_2, rotx_3], axis=2) 
    
    
    rotMat = torch.mat(torch.mat(xmat, ymat), zmat)
    
    return rotmat

    


# In[33]:


# #SFM Learner-Change accordingly
# def euler2mat(angle):
#     """Convert euler angles to rotation matrix.
#      Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
#     Args:
#         angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
#     Returns:
#         Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]    
#     """
#     B = angle.size(0)
#     x, y, z = angle[:,0], angle[:,1], angle[:,2]

#     cosz = torch.cos(z)
#     sinz = torch.sin(z)

#     zeros = z.detach()*0
#     ones = zeros.detach()+1
#     zmat = torch.stack([cosz, -sinz, zeros,
#                         sinz,  cosz, zeros,
#                         zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

#     cosy = torch.cos(y)
#     siny = torch.sin(y)

#     ymat = torch.stack([cosy, zeros,  siny,
#                         zeros,  ones, zeros,
#                         -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

#     cosx = torch.cos(x)
#     sinx = torch.sin(x)

#     xmat = torch.stack([ones, zeros, zeros,
#                         zeros,  cosx, -sinx,
#                         zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

#     rotMat = xmat @ ymat @ zmat
#     return rotMat


# In[ ]:


# def rot2euler(R):
#     sy = torch.sqrt(R[:,0,0] * R[:,0,0] +  R[:,1,0] * R[:,1,0])
#     eps = torch.tensor(1e-6, shape=)


# In[5]:


#Change Accordingly-May not need this
def axis_angle_to_rotation_matrix(axis,angle):
    B=angle.size(0)
    z=angle[:,2]
    zeros=z.detach()*0
    ones = zeros.detach()+1
    
    Mat1=torch.cat([zeros,-torch.expand_dims(torch.expand_dims(axis[:,2],-1),-1), torch.expand_dims(torch.expand_dims(axis[:,1],-1),-1)], axis=2)
    Mat2=torch.cat([zeros, zeros, -torch.expand_dims(torch.expand_dims(axis[:,0],-1),-1)],axis=2)
    Mat3=torch.cat([zeros,zeros,zeros],axis=2)
    
    Mat=torch.cat([Mat1,Mat2,Mat3],axis=1)
    
    cp_axis=Mat-torch.transpose(Mat,perm=[0,2,1])
    
    RotMat=torch.eye(3, batch_shape=[B])+torch.sin(angle)*cp_axis+(ones-torch.cose(angle))*torch.matmaul(cp_axis, cp_axis)
    
    return RotMat


# In[11]:


#Original-Rewritten
def pose_vec2mat(vec,format):
    
    batch_size = list(vec.shape())
    translation = vec[:, :3]
    translation =  torch.unsqueeze(-1)
    
    if format == 'euler':
        rx = vec[:, 3:4]
        ry = vec[:, 4:5]
        rz = vec[:, 5:6]
        rot_mat = euler2mat(rz, ry, rx)
        rot_mat = torch.squeeze(1)
        
    elif format == 'angleaxis':
        axis = vec[:, 3:]
        angle = torch.unsqueeze(torch.norm(axis, axis=1), -1)
        axis = axis/angle
        angle = torch.unsqueeze(-1)
        rot_mat = axis_angle_to_rotation_matrix(axis, angle)
        
    filler = torch.tensor([0,0,0,1]).reshape(1,1,4)
    filler = torch.repeat_interleave(filler, repeats = batch_size, dim=0)
    transform_mat = torch.cat([rot_mat, translation], axis = 2)
    transform_mat = torch.cat([transform_mat, filler], axis =1)
    
    return transform_mat
        


# In[3]:


# #Modify according to axis_angle_to_rotation_matrix -Probably need changing
# def pose_vec2mat(vec, format):
#     """
#     Convert 6DoF parameters to transformation matrix.
#     Args:s
#         vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
#     Returns:
#         A transformation matrix -- [B, 3, 4]
#     """
#     translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
#     rot = vec[:,3:]
# #     filler=torch.tensor([[0, 0, 0, 1]]).type(dtype)
#     if format == 'euler':
#         rot_mat = euler2mat(rot)  # [B, 3, 3]
    
#     elif format == 'angleaxis':
#         angle=torch.expand_dims(torch.norm(rot, dim=1),-1)
#         rot_mat = axis_angle_to_rotation_matrix(rot,angle)  # [B, 3, 3]
#     transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
#     return transform_mat


# In[14]:


##Original-Rewritten
def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

        Args:
        depth: [batch, height, width]
        pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
        intrinsics: camera intrinsics [batch, 3, 3]
        is_homogeneous: return in homogeneous coordinates
        Returns:
        Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    
    batch, height, width = list(depth.shape)
    depth = depth.reshape(batch, 1, -1)
    pixel_coords = pixel_coords.reshape(batch, 3, -1)
    cam_coords = torch.mul(torch.inverse(intrinsics), pixel_coords) * depth
    
    if is_homogeneous:
        ones = torch.ones(batch,1, height*width)
        cam_coords = torch.cat((cam_coords, ones),1)
    cam_coords = torch.reshape(batch, -1, height, width)
    
    return cam_coords
    


# In[8]:


# #SFM Learner-Needs changing
# def pixel2cam(depth, pixel_coords,intrinsics_inv):
#     global pixel_coords
#     """Transform coordinates in the pixel frame to the camera frame.
#     Args:
#         depth: depth maps -- [B, H, W]
#         intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
#         is_homogeneous: return homogeneous coordinates
#     Returns:
#         array of (u,v,1) cam coordinates -- [B, 3, H, W]
#     """
#     b, h, w = depth.size()
#     if (pixel_coords is None) or pixel_coords.size(2) < h:
#         set_id_grid(depth)
#     current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
#     cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
#     return cam_coords * depth.unsqueeze(1)


# In[2]:


def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

        Args:
        cam_coords: [batch, 4, height, width]
        proj: [batch, 4, 4]
        Returns:
        Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = list(cam_coords.shape)
    unnormalized_pixel_coords = torch.mul(proj, cam_coords)
    x_u = unnormalized_pixel_coords[0:-1, 0:1, 0:-1]
    y_u = unnormalized_pixel_coords[0:-1, 1:1, 0:-1]
    z_u = unnormalized_pixel_coords[0:-1, 2:1, 0:-1]
    x_n = x_u/(z_u + 1e-10)
    y_n = y_u/(z_u + 1e-10)
    pixel_coords = torch.cat((x_n, y_n), 1)
    pixel_coords = pixel_coords.reshape(batch, 2, height, width)
    z_u = z_u.reshape(batch, height, width, 1)
    
    return pixel_coords.permute(0,2,3,1), z_u

    


# In[1]:


# #SFM Learner-Needs changing
# def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr):
#     """Transform coordinates in the camera frame to the pixel frame.
#     Args:
#         cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
#         proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
#         proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
#     Returns:
#         array of [-1,1] coordinates -- [B, 2, H, W]
#     """
#     b, _, h, w = cam_coords.size()
#     cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
#     if proj_c2p_rot is not None:
#         pcoords = proj_c2p_rot @ cam_coords_flat
#     else:
#         pcoords = cam_coords_flat

#     if proj_c2p_tr is not None:
#         pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
#     X = pcoords[:, 0]
#     Y = pcoords[:, 1]
#     Z = pcoords[:, 2].clamp(min=1e-3)

#     X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
#     Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

#     pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
#     return pixel_coords.reshape(b,2,h,w)


# In[2]:


def meshgrid(batch, height, width):
    
    """Construct a 2D meshgrid.
  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
    x_t=torch.matmul(torch.ones([height,1]), torch.transpose(torch.unsqueeze(torch.linspace(-1,1,width),1),1,0))
    y_t=torch.matmul(torch.unsqueeze(torch.linspace(-1,1,height),1), torch.ones([1, width]))
    x_t = (x_t + 1.0) * 0.5 * torch.tensor(width - 1).float()
    y_t = (y_t + 1.0) * 0.5 * torch.tensor(height - 1).float()
    coords=torch.stack((x_t,y_t),0)
    coords=torch.repeat_interleave(torch.unsqueeze(coords,0), batch, dim=0) 


# In[15]:


#Dims checked, wont need changing
def inverse_warp(img, depth, pose, intrinsics, format='eular'):
    """Inverse warp a source image to the target image plane based on projection
    Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 6], in the
          order of tx, ty, tz, rx, ry, rz
    intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
    """

    batch, height, width,_ = img.size()

    #Pose estimate:
    pose = pose_vec2mat(pose,format)

    #pixel_grid
    pixel_coords = meshgrid(batch, height, width)

    #pixel_coords->camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)

    #Intrinsic Matrix-4x4
    #TODO: Can you make it 3x4
    filler=torch.tensor([[[0,0,0,1]]])
    filler=torch.repeat_interleave(filler, batch, dim=0)
    intrinsics=torch.cat((intrinsics, torch.zeros([batch,3,1])), 2)
    intrinsics=torch.cat((intrinsics, filler), 1)

    #Project=Target-Cam->source-pixel-frame
    proj_tgt_cam_2_src_pixel=torch.matmul(intrinsics, pose)
    src_pixel_coords, src_depth=cam2pixel(cam_coords, proj_tgt_cam_2_src_pixel)

    rigid_flow=src_pixel_coords-pixel_coords[:,0:2,:,:].permute(0,2,3,1)
    output_img,wmask = bilinear_sampler(img, src_pixel_coords)
    return output_img, wmask, rigid_flow


# In[16]:


#Not Needed
def extract_image(img, src_pixel_coords,src_depth):
    
    batch, height, width, _=img.size()
    out_img=torch.zeros([batch, height, width, 3])
    out_depth=torch.zeros([batch,height, width, 1])
    
    return out_img, out_depth


# In[ ]:


# #skipped it
# def optflow_wrap:


# In[2]:


def billinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
    Returns:
    A new sampled image [batch, height_t, width_t, channels]
    """
    def _repeat(x, n_repeats):
        rep=torch.transpose((torch.unsqueeze(torch.ones([n_repeats]),1)),1,0).float()
        x=torch.matmul(torch.reshape(x, (-1,1)),rep)
        
        return x.reshape(x,[-1])
    
    #Was originally with tf.name_scope('image sampling')
    def forward():
        coords_x, coords_y = torch.split(coords, [1,1], dim=3).float()
        inp_size=img.shape
        coord_size=coords.shape
        out_size=coords.shape
        out_size[3]=imgs.shape[3]
        
        x0=torch.floor(coords_x)
        x1=x0+1
        y0=torch.floor(coords_y)
        y1=y0+1
        
        y_max=(imgs.shape[1]-1).float()
        x_max=(imgs.shape[2]-1).float()
        zeros=torch.zeros([1]).float()
        
        x0_safe=torch.clamp(x0, zero, x_max)
        y0_safe=torch.clamp(y0, zero, y_max)
        x1_safe=torch.clamp(x1, zero, x_max)
        y1_safe=torch.clamp(y1, zero, y_max)
        
        wt_x0=(x1-coords_x)*torch.equal(x0, x0_safe)
        wt_x1=(coords_x-x0)*torch.equal(x1, x1_safe)
        wt_y0=(y1-coords_y)*torch.equal(y0, y0_safe)
        wt_y1=(coords_y-y0)*torch.equal(y1, y1_safe)
        
        dim1=tensor.double(inp_size[2]*inp_size[1])
        dim2=tensor.double(inp_size[2])
        
        
        base=torch.reshape(_repeat(coord_size[0]*dim1*coord_size[1]*coord_size[2]),[out_size[0], out_size[1], out_size[2], 1])
        base_y0=base+y0_safe*dim2
        base_y1=base+y1_safe*dim2
        idx00=torch.reshape((x0_safe+base_y0),-1)
        idx01=x0_safe+base_y1
        idx10=x1_safe+base_y0
        idx11=x1_safe+base_y1
        
        imgs_flat=torch.reshape(imgs, torch.stack((torch.tensor(-1),torch.tensor(inp_size[3])))).float()
        
        im00=torch.reshape(imgs_flat[idx00],out_size)
        im01=torch.reshape(imgs_flat[idx01],out_size)
        im10=torch.reshape(imgs_flat[idx10],out_size)
        im11=torch.reshape(imgs_flat[idx11],out_size)
        
        w00=wt_x0*wt_y0
        w01=wt_x0*wt_y1
        w10=wt_x1*wt*y_0
        w11=wt_x1*wt_y1
        
        output=torch.add(torch.tensor([w00*im00, w01*im01]),torch.tensor([w10 * im10, w11 * im11]))
        
        return output
        


# In[2]:


# #Skipped it
# def depth_optlfow():
#     #left


# In[4]:


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    u[u>minu] = minu
    u[u<maxu] = maxu

    v[v>minv] = minv
    v[v<maxv] = maxv

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    #print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


# In[6]:


def depth_plasma(depth):
    cmap = plt.get_cmap('plasma')

    rgb_depth = cmap(depth/np.max(depth))
    return np.float32(rgb_depth)


# In[7]:


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


# In[8]:


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


# In[1]:


# #Skipped it
# def detect_reflection(img):
    


# In[ ]:




