import numpy as np
import matplotlib.pyplot as plt
import cv2

##########################################################################################################################

## Part1 Calibration

# Features matching with SIFT
def SiftMatches(img_l_gray, img_r_gray, img_l, img_r):
    sift =cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img_l_gray, None)
    kp_r, des_r = sift.detectAndCompute(img_r_gray, None)
    # matcher brute force
    bf = cv2.BFMatcher()
    found_matches = bf.match(des_l, des_r)
    found_matches = sorted(found_matches, key= lambda x : x.distance)
    matches = found_matches[0:100]

    matched_points = []
    for i, j in enumerate(matches):
        pt_l = kp_l[j.queryIdx].pt
        pt_r = kp_r[j.trainIdx].pt
        matched_points.append([pt_l[0], pt_l[1], pt_r[0], pt_r[1]])
    matched_points = np.array(matched_points).reshape(-1,4)

    matched_img = cv2.drawMatches(img_l, kp_l, img_r, kp_r, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_points, matched_img

# RANSAC
def RansacMatchedPoints(matched_points):
    # Initiating the values
    threshold = 0.02
    length_threshold = 0
    indices_final = []
    F_final = 0

    for i in range(0, 1000):
        ind = []
        size =matched_points.shape[0]
        rand_ind =np.random.choice(size, size=8)
        rand_matched_points = matched_points[rand_ind, :]
        fund_matrix = EstimateFundamentalMatrix(rand_matched_points)
        
        for j in range(size):
 
            x1,x2 = matched_points[j][0:2], matched_points[j][2:4]
            xl_matrix_T=np.array([x1[0], x1[1], 1]).T
            xr_matrix=np.array([x2[0], x2[1], 1])
            error = np.dot(xl_matrix_T, np.dot(fund_matrix, xr_matrix))

            if error < threshold:
                ind.append(j)

        if len(ind) > length_threshold:
            length_threshold = len(ind)
            indices_final = ind
            F_final= fund_matrix


    matched_final = matched_points[indices_final, :]
    
    
    return F_final, matched_final

# normalizing the values
def normalization(uv):

    uv_prime = np.mean(uv, axis=0)
    u_prime ,v_prime = uv_prime[0], uv_prime[1]

    u_cap = uv[:,0] - u_prime
    v_cap = uv[:,1] - v_prime

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_prime],[0,1,-v_prime],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

# computing the fundamental matrix
def EstimateFundamentalMatrix(matched_f):

    x1 = matched_f[:,0:2]
    x2 = matched_f[:,2:4]

    x1, T1 = normalization(x1)
    x2, T2 = normalization(x2)

            
    A = np.zeros((len(x1),9))
    
    
    for i in range(0, len(x1)):
        u_l,v_l = x1[i][0], x1[i][1]
        u_r,v_r = x2[i][0], x2[i][1]
        A[i] = np.array([u_l*u_r, u_r*v_l, u_r, v_r*u_l, v_r*v_l, v_r, u_l, v_l, 1])

    U, E_values, E_vectors = np.linalg.svd(A, full_matrices=True)
    F = E_vectors.T[:, -1]
    F = F.reshape(3,3)

    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2,2] = 0
    F = np.dot(u, np.dot(s, vt))

    F = np.dot(T2.T, np.dot(F, T1))
    return F

# Computing essential matrix
def EssentialMatrix(K1, K2, F):
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_corrected = np.dot(U,np.dot(np.diag(s),V))
    return E_corrected

def ExtractCameraPose(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C

def Compute3DPoints(K1, K2, matched_points, R2, C2):
    points_3d = []
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    I = np.identity(3)
    P1 = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))

    for i in range(len(C2)):
        x1 = matched_points[:,0:2].T
        x2 = matched_points[:,2:4].T

        P2 = np.dot(K2, np.dot(R2[i], np.hstack((I, -C2[i].reshape(3,1)))))

        X = cv2.triangulatePoints(P1, P2, x1, x2)  
        points_3d.append(X)
    return points_3d

def ComputePositiveZCount(points_3D, R, C):
    I = np.identity(3)
    Extrinsic_matrix = np.dot(R, np.hstack((I, -C.reshape(3,1))))
    Extrinsic_matrix = np.vstack((Extrinsic_matrix, np.array([0,0,0,1]).reshape(1,4)))
    n_positiveZ = 0
    for i in range(points_3D.shape[1]):
        X_world = points_3D[:,i]
        X_world = X_world.reshape(4,1)
        X_camera = np.dot(Extrinsic_matrix, X_world)
        X_camera = X_camera / X_camera[3]
        z = X_camera[2]
        if z > 0:
            n_positiveZ += 1

    return n_positiveZ

# Rotation and translation matrices
def ComputeFinal_R_T(points_3d,R,C):
    z_c_l = []
    z_c_r = []
    
    Rl = np.identity(3)
    Tl = np.zeros((3,1))
    for i in range(len(points_3d)):
        pts3D = points_3d[i]
        pts3D = pts3D/pts3D[3, :]  
    
        z_c_r.append(ComputePositiveZCount(pts3D, R[i], T[i]))
        z_c_l.append(ComputePositiveZCount(pts3D, Rl, Tl))
    
    z_c_r = np.array(z_c_r)
    z_c_l = np.array(z_c_l)
    
    count_threshold = int(points_3d[0].shape[1] / 2)
    idx = np.intersect1d(np.where(z_c_l > count_threshold), np.where(z_c_r > count_threshold))
    R_final = R[idx[0]]
    T_final = T[idx[0]]
    return R_final,T_final

##################################################################################################################################################

## Rectification

# computing epipolarlines
def ComputeEpipolarLines(set1, set2, F, image_l, image_r):
    
    lines1, lines2 = [], []
    img_ep_l = image_l.copy()
    img_ep_r = image_r.copy()

    for i in range(set1.shape[0]):
        x1 = np.array([set1[i,0], set1[i,1], 1]).reshape(3,1)
        x2 = np.array([set2[i,0], set2[i,1], 1]).reshape(3,1)

        line1 = np.dot(F.T, x2)
        line2 = np.dot(F, x1)
        lines1.append(line1)        
        lines2.append(line2)

        x1_min = 0
        x1_max = image_r.shape[1] -1
        y1_min = -line1[2]/line1[1]
        y1_max = -line1[2]/line1[1]

        x2_min = 0
        x2_max = image_l.shape[1] - 1
        y2_min = -line2[2]/line2[1]
        y2_max = -line2[2]/line2[1]

        cv2.circle(img_ep_l, (int(set1[i,0]),int(set1[i,1])), 10, (0,0,255), -1)
        img_ep_l = cv2.line(img_ep_l, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0, int(i*2.55)), 2)

        cv2.circle(img_ep_r, (int(set2[i,0]),int(set2[i,1])), 10, (0,0,255), -1)
        img_ep_r = cv2.line(img_ep_r, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, int(i*2.55)), 2)

    
    epipolar_img = np.concatenate((img_ep_l, img_ep_r), axis = 1)
    epipolar_img_before = cv2.resize(epipolar_img, (1920, 1080))
    
    return lines1, lines2,epipolar_img_before

# Drawing epipolarlines on the rectified images
def Compute_RectImg_EpiPolarLines(img_l,img_r, matched_final):
        
    uv_l, uv_r = matched_final[:,0:2], matched_final[:,2:4]

    height1, width1 = img_l.shape[:2]
    height2, width2 = img_r.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(uv_l), np.float32(uv_r), F, imgSize=(width1, height1))
    
    print(" Homography for left image h1: ")
    print('\n'.join(['   '.join(['{:4}'.format(element) for element in eachrow]) for eachrow in H1]))
    print("Homography for right image h2: ")
    print('\n'.join(['   '.join(['{:4}'.format(element) for element in eachrow]) for eachrow in H2]))
     
    img1_rectified = cv2.warpPerspective(img_l, H1, (width1, height1))
    img2_rectified = cv2.warpPerspective(img_r, H2, (width2, height2))
     
    uv_l_rectified = cv2.perspectiveTransform(uv_l.reshape(-1, 1, 2), H1).reshape(-1,2)
    uv_r_rectified = cv2.perspectiveTransform(uv_r.reshape(-1, 1, 2), H2).reshape(-1,2)
     
    H1_inv = np.linalg.inv(H1)
    H2_T_inv =  np.linalg.inv(H2.T)
    F_rectified = np.dot(H2_T_inv, np.dot(F, H1_inv))
     
    lines1_rectified, lines2_rectified,epipolar_img = ComputeEpipolarLines(uv_l_rectified, uv_r_rectified, F_rectified, img1_rectified, img2_rectified)
     
    imgl_rectified = cv2.resize(img1_rectified, (int(img1_rectified.shape[1] / 4), int(img1_rectified.shape[0] / 4)))
    imgr_rectified= cv2.resize(img2_rectified, (int(img2_rectified.shape[1] / 4), int(img2_rectified.shape[0] / 4)))
     
    imgl_rectified = cv2.cvtColor(imgl_rectified, cv2.COLOR_BGR2GRAY)
    imgr_rectified = cv2.cvtColor(imgr_rectified, cv2.COLOR_BGR2GRAY)   
    
    
    return imgl_rectified,imgr_rectified,lines1_rectified,lines2_rectified,epipolar_img
    

##################################################################################################################################################

## Correspondence
# Calculating sum of square distances
def Compute_SSD(pixel_vals_1, pixel_vals_2):
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum((pixel_vals_1 - pixel_vals_2)**2)

def CompareBlock(y, x, block_left, right_array, block_size, x_search_block_size, y_search_block_size):
    x_min = max(0, x - x_search_block_size)
    x_max = min(right_array.shape[1], x + x_search_block_size)
    y_min = max(0, y - y_search_block_size)
    y_max = min(right_array.shape[0], y + y_search_block_size)
    
    first = True
    min_ssd = None
    min_index = None

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            block_right = right_array[y: y+block_size, x: x+block_size]
            ssd = Compute_SSD(block_left, block_right)
            if first:
                min_ssd = ssd
                min_index = (y, x)
                first = False
            else:
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_index = (y, x)

    return min_index


def ComputeCorrespondence(imgl, imgr):

    block_size = 15 
    x_search_block_size = 50 
    y_search_block_size = 1
    h, w = imgl.shape
    disparity_map = np.zeros((h, w))

    for y in range(block_size, h-block_size):
        for x in range(block_size, w-block_size):
            block_left = imgl[y:y + block_size, x:x + block_size]
            index = CompareBlock(y, x, block_left, imgr, block_size, x_search_block_size, y_search_block_size)
            disparity_map[y, x] = abs(index[1] - x)
    
    disparity_map_unscaled = disparity_map.copy()
    # Scaling the disparity map
    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)

    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            disparity_map[i][j] = int((disparity_map[i][j]*255)/(max_pixel-min_pixel))
    
    return disparity_map,disparity_map_unscaled

###################################################################################################################################################

## Compute Depth Image
def ComputeDepthMap(baseline, focal_length, image):

    depth_map = np.zeros((image.shape[0], image.shape[1]))
    depth = np.zeros((image.shape[0], image.shape[1]))

    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth_map[i][j] = 1/image[i][j]
            depth[i][j] = (focal_length* baseline)/image[i][j]


    return depth_map, depth

###################################################################################################################################################
# camera parameters from calib.txt
cam0 = np.array([[1758.23, 0, 829.15], [0, 1758.23, 552.78], [0, 0, 1]])
cam1 = np.array([[1758.23, 0, 829.15], [0, 1758.23, 552.78], [0, 0, 1]])
baseline = 97.99

focal_length = cam0[0,0]
img_l = cv2.imread("im0.png")
img_l_gray = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

img_r = cv2.imread("im1.png")
img_r_gray = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
matched_points,img_matched= SiftMatches(img_l_gray, img_r_gray, img_l, img_r)


F, matched_final = RansacMatchedPoints(matched_points)
print("The fundamental matrix:" ,F)
E = EssentialMatrix(cam0, cam1, F)
print("The Essential matrix: ", E)
R,T= ExtractCameraPose(E)


points_3d = Compute3DPoints(cam0, cam1, matched_final, R, T)

R_final,T_final= ComputeFinal_R_T(points_3d, R, T)
print("Rotation matrix: ", R_final)
print("Translation matrix: ", T_final)

imgl_rectified,imgr_rectified,lines1_rectified,lines2_rectified,epipolar_img  = Compute_RectImg_EpiPolarLines(img_l, img_r, matched_final)


disparity_map,disparity_map_unscaled= ComputeCorrespondence(imgl_rectified, imgr_rectified)

depth_map, depth= ComputeDepthMap(baseline, focal_length, disparity_map_unscaled)

plt.imshow(epipolar_img)
plt.savefig("chess_epipolar_img.png")
plt.imshow(disparity_map, cmap='hot', interpolation='nearest')
plt.savefig("chess_disparity_map_heat.png")
plt.imshow(disparity_map, cmap='gray', interpolation='nearest')
plt.savefig("chess_disparity_map_gray.png")
plt.imshow(depth_map, cmap='hot', interpolation='nearest')
plt.savefig("chess_depth_map_heat.png")
plt.imshow(depth_map, cmap='gray', interpolation='nearest')
plt.savefig("chess_depth_map_gray.png")
print("Stereo Vision Done!!!")