import numpy as np
from skimage import filters, feature

def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    a = 0.06
    thresh = 0.005
    stepsize = 2
    min_distance = 3
    
    filtered_image = filters.gaussian(image, sigma=0.1)
    I_x = filters.sobel_v(filtered_image)
    I_y = filters.sobel_h(filtered_image)

    Ixx = filters.gaussian(I_x ** 2, sigma=0.1)
    Ixy = filters.gaussian(I_y * I_x, sigma=0.1)
    Iyy = filters.gaussian(I_y ** 2, sigma=0.1)
    R_score = np.zeros_like(image)

    #caculate R matrix
    for y in range(0, image.shape[0] - feature_width, stepsize):
        for x in range(0, image.shape[1] - feature_width, stepsize):
            Wxx = np.sum(Ixx[y:y + feature_width + 1, x:x + feature_width + 1])
            Wyy = np.sum(Iyy[y:y + feature_width + 1, x:x + feature_width + 1])
            Wxy = np.sum(Ixy[y:y + feature_width + 1, x:x + feature_width + 1])

            det = (Wxx * Wyy) - (Wxy ** 2)
            trace = Wxx + Wyy
            R = det - a * (trace ** 2)

            if R > thresh:
                R_score[y + feature_width // 2, x + feature_width // 2] = R

    #non-maximal suppression
    pts_cords = feature.peak_local_max(R_score, min_distance=min_distance, threshold_abs=thresh)
    xs = pts_cords[:, 1]
    ys = pts_cords[:, 0]

    return xs, ys

def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    #assert feature indecies as integers
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    features = np.zeros((len(x), 4, 4, 8))
    
    # helper function that returns magnitude and angle of gradient
    def get_gradient (image, sigma):
        img_blur = filters.gaussian(image, sigma)
        dx = filters.sobel_v(img_blur)
        dy = filters.sobel_h(img_blur)
        magnitude = np.hypot(dx,dy)
        angle = np.arctan2(dy,dx)
        angle[angle < 0] += 2 * np.pi
        return magnitude, angle   
    
    gradmag, gradangle = get_gradient(image, 0.1)

    
    # constructing and looping on a feature window for each interest keypoint (x,y)
    for p, (xf, yf) in enumerate(zip(x,y)):
        rows = (yf - feature_width//2 - 1, yf + feature_width//2 - 1)
        cols = (xf - feature_width//2 - 1, xf + feature_width//2 - 1)
        
        if rows[0] < 0:
            rows = (0, feature_width - 1)
        if rows[1] > image.shape[0] - 1:
            rows = (image.shape[0] - feature_width, image.shape[0] - 1)
        if cols[0] < 0:
            cols = (0, feature_width - 1)
        if cols[1] > image.shape[1] - 1:
            cols = (image.shape[1] - feature_width, image.shape[1] - 1)
        
        #use our rows and columns dimensions to create gradient windows (feature vector)
        gradmag_window = gradmag[rows[0] : rows[1], cols[0] : cols[1]]
        gradangle_window = gradangle[rows[0] : rows[1], cols[0] : cols[1]]
        
        #we will apply a gaussian filter on the window so that points closer to the keypoint contributes more to the histogram weights
        gradmag_window = filters.gaussian(gradmag_window, 0.2)
        gradangle_window = filters.gaussian(gradangle_window, 0.2)
        
        #estimate feature orientation
        orientations, bin_angle  = np.histogram(gradangle_window.reshape(-1), bins = 36, range = (0, 2*np.pi),
                                    weights = gradmag_window.reshape(-1))
        dominant_angle = bin_angle[np.argmax(orientations)]
        
        '''
        if dominant_angle >= 0 and dominant_angle <= np.pi/2:
            rot_angle = np.pi/2 - dominant_angle
            gradangle_window += rot_angle
        elif dominant_angle > np.pi/2 and dominant_angle <= 2 * np.pi:
            rot_angle = dominant_angle - np.pi/2
            gradangle_window -= rot_angle
            
        gradangle_window[gradangle_window < 0] += 2 * np.pi
        '''
        
        #splitting the window into 4x4 cells
        for i in range(feature_width//4):
            for j in range(feature_width//4):
                gradmag_cell = gradmag_window[i * feature_width//4 : (i + 1) * feature_width//4 - 1, j : (j + 1) * feature_width//4 - 1]
                gradangle_cell = gradangle_window[i * feature_width//4 : (i + 1) * feature_width//4 - 1, j : (j + 1) * feature_width//4 - 1]
                features[p, i, j] = np.histogram(gradangle_cell.reshape(-1), bins = 8, range = (0, 2*np.pi),
                                                 weights = gradmag_cell.reshape(-1))[0]
                
    #reshaping the features container (keypoints length, 128 feature-vector)
    features = features.reshape(len(x),-1)
    
    #normalization
    norm = np.linalg.norm(features, axis = 1).reshape(-1,1)
    features = features / norm
    
    threshold = 0.2
    features[features > threshold] = threshold
    
    norm = np.linalg.norm(features, axis = 1).reshape(-1,1)
    features = features / norm
    
    features = features ** 0.69
   

    return features

def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    match1 = []
    confidences = []
    for i in range(im1_features.shape[0]):
            ##if the vector is all zeros, we will not consider matching
            if np.std(im1_features[i, :]) != 0:
                d = np.sqrt(np.square(np.subtract(
                    im1_features[i, :], im2_features)).sum(axis=1))
                orders = np.argsort(d).tolist()
                ratio = d[orders[0]] / d[orders[1]]
                percent = abs(ratio - 1)
                if ratio<=0.8:
                    confidences.append(percent)
                    match1.append((i, orders[0]))

    """for i in range(im2_features.shape[0]):
            ##if the vector is all zeros, we will not consider matching
            if np.std(im2_features[i, :]) != 0:
                d = im1_features - im2_features[i, :]
                d = np.linalg.norm(d, axis=1)
                orders = np.argsort(d).tolist()
                ratio = d[orders[0]] / d[orders[1]]
                percent = abs(ratio-1)
                if percent>=0.8:
                    match2.append((orders[0], i))
                    confidences.append(percent)

        ##find good matches in rotation tests both ways
    matches = list(set(match1).intersection(set(match2)))"""
    matches = np.asarray(match1)
    confidences = np.asarray(confidences)
    
    return matches, confidences