import cv2
import numpy as np
import functools
import random


def draw_a_spot(img, key):
    y, x = np.round(key.pt)
    x = int(x)
    y = int(y)
    img[x - 1 : x + 2, y - 1: y + 2, 0] = 255
    img[x - 1 : x + 2, y - 1: y + 2, 1 : -1] = 0

def draw_a_keypoint(img, key):
    x, y = np.round(key.pt).astype('int')
    cv2.circle(img, (x, y), round(key.size), 255, 2)
    x2 = round(np.sin(360 - key.angle) * key.size) + x
    y2 = round(np.cos(360 - key.angle) * key.size) + y
    cv2.line(img, (x,y), (x2,y2), 255, 2)


def generate_pyramid(image, sigma, num_intervals, assumed_blur):
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    image =  cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
    num_octaves = int(round(np.log(min(image.shape)) / np.log(2) - 1))
    k = 2 ** (1. / num_intervals)
    expect = np.zeros(num_intervals + 3)
    sigmas = np.zeros(num_intervals + 3) 
    expect[0] = sigma
    sigmas[0] = sigma
    for i in range(1, num_intervals + 3):
        expect[i] = (k ** i) * sigma
        sigmas[i] = np.sqrt(expect[i] ** 2 - expect[i-1] ** 2)
    gaussian_images = []
    dog_images = []
    for i in range(num_octaves):
        gaussian_images_in_octave = [image]
        dog_images_in_octave = []
        for j in range(1, len(sigmas)):
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigmas[j], sigmaY=sigmas[j])
            gaussian_images_in_octave.append(image)
            dog = gaussian_images_in_octave[j] - gaussian_images_in_octave[j - 1]
            dog_images_in_octave.append(dog)
        gaussian_images.append(np.array(gaussian_images_in_octave))
        dog_images.append(np.array(dog_images_in_octave))
        octave_base = gaussian_images_in_octave[-3]
        image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
    # return np.array(gaussian_images, dtype=object), np.array(dog_images, dtype=object)
    return gaussian_images, dog_images

def calc_keypoint(i, j, image_index, octave_index, num_intervals, pixel_cube, sigma, contrast_threshold, eigenvalue_ratio=10):
    gradient = 0.5 * np.array([pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0], 
                     pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1], pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1]])
    hessian = np.array([[pixel_cube[1, 1, 2] - 2 * pixel_cube[1, 1, 1] + pixel_cube[1, 1, 0],
                           0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0]),
                             0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0])], 
                    [0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0]),
                      pixel_cube[1, 2, 1] - 2 * pixel_cube[1, 1, 1] + pixel_cube[1, 0, 1],
                        0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])],
                    [0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0]),
                      0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1]),
                        pixel_cube[2, 1, 1] - 2 * pixel_cube[1, 1, 1] + pixel_cube[0, 1, 1]]])
    extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
    if (abs(extremum_update) * num_intervals < contrast_threshold).any():
        return None
    xy_hessian = hessian[:2, :2]
    xy_hessian_trace = np.trace(xy_hessian)
    xy_hessian_det = np.linalg.det(xy_hessian)
    if xy_hessian_det < 0:
        return None
    if eigenvalue_ratio * (xy_hessian_trace ** 2) >= ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
        return None
    keypoint = cv2.KeyPoint()
    keypoint.pt = ((j * (2 ** octave_index), i * (2 ** octave_index)))
    keypoint.octave = octave_index + image_index * (2 ** 8)
    keypoint.size = sigma * (2 ** (image_index / np.float32(num_intervals))) * (2 ** (octave_index + 1)) 
    return keypoint

def calc_orientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index+1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)
    point = np.round(keypoint.pt / np.float32(2 ** octave_index)).astype("int")
    temp = point[0]
    point[0] = point[1]
    point[1] = temp
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            region = point + np.array([i, j])
            if np.min(region) > 0 and region[0] < image_shape[0] - 1 and region[1] < image_shape[1] - 1:
                dx = gaussian_image[region[0], region[1] + 1] - gaussian_image[region[0], region[1] - 1]
                dy = gaussian_image[region[0] - 1, region[1]] - gaussian_image[region[0] + 1, region[1]]
                gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                weight = np.exp(weight_factor * (i ** 2 + j ** 2)) 
                histogram_index = int(round(gradient_orientation * num_bins / 360.))
                raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = []
    for i, value in enumerate(smooth_histogram):
        if value > max(smooth_histogram[i-1], smooth_histogram[(i+1) % smooth_histogram.size]):
            orientation_peaks.append(i)
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            orientation = 360. - peak_index * 360. / num_bins
            if abs(orientation - 360.) < 1e-7:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def find_keypoint(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid
    """
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
    keypoints = []

    for octave_index, dog_image in enumerate(dog_images):
        dog_images_in_octave = np.array(dog_image)
        for image_index in range(1, dog_images_in_octave.shape[0] - 1):
            slice = dog_images_in_octave[image_index-1 : image_index + 2, :, :]
            for i in range(image_border_width, slice.shape[1] - image_border_width):
                for j in range(image_border_width, slice.shape[2] - image_border_width):
                    cube = slice[:, i-1:i+2, j-1:j+2]
                    neighbours = np.delete(cube.flatten(), 13)
                    if abs(cube[1][1][1]) >= threshold and (np.all(cube[1][1][1] > neighbours) or np.all(cube[1][1][1] < neighbours)):
                        keypoint = calc_keypoint(i, j, image_index + 1, octave_index, num_intervals, cube, sigma, contrast_threshold, image_border_width)
                        if keypoint is not None:
                            orientations = calc_orientations(keypoint, octave_index, gaussian_images[octave_index][image_index])
                            for keypoint_with_orientation in orientations:
                                keypoints.append(keypoint_with_orientation)
    if len(keypoints) < 2:
        return keypoints
    keypoints.sort(key=functools.cmp_to_key(compareKeypoints))
    seen = {}
    for keypoint in keypoints:
        key = (keypoint.pt, keypoint.size, keypoint.angle)
        if key not in seen:
            seen[key] = keypoint
    unique_keys =  list(seen.values())
    for keypoint in unique_keys:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
    return unique_keys

def calc_descriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint
    """
    descriptors = []
    bins_per_degree = num_bins / 360.
    weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)

    for keypoint in keypoints:
        octave = keypoint.octave & 255
        image = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        gaussian_image = gaussian_images[octave + 1][image]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c = np.zeros([2, 2, 2])
            for i in range(2):
                for j in  range(2):
                    for k in range(2):
                        c[i][j][k] = magnitude * (row_fraction ** i) * ((1 - row_fraction) **(1 - i)) * (col_fraction ** j) * ((1 - col_fraction) **(1 - j)) * (orientation_fraction ** k) * ((1 - orientation_fraction) **(1 - k))
                        histogram_tensor[row_bin_floor + 1 + i, col_bin_floor + 1 + j, (orientation_bin_floor + k) % num_bins] += c[i][j][k]

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector = descriptor_vector.clip(0, 255)
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint
    """
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1][layer]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')

def quick_calc(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    image = np.sum(image, 2)
    image = image.astype('float32')
    image *= (255 / np.max(image))
    gaussian_images, dog_images = generate_pyramid(image, sigma, num_intervals, assumed_blur)
    keypoints = find_keypoint(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    descriptors = calc_descriptors(keypoints, gaussian_images)
    return keypoints, descriptors

class SIFT(object):
    def __init__(self, sigma=1.6, num_intervals=3, border = 5):
        """
        Implement Scale-Invariant Feature Transform(SIFT) algorithm
        :param kwargs: other hyperparameters, such as sigma, blur ratio, border, etc.
        """
        self.sigma = sigma
        self.num_intervals = num_intervals
        self.border = border

    # =========================================================================================================
    # TODO: you can add other functions here or files in this directory
    # =========================================================================================================    


    def out(self, img):
        """
        Implement Scale-Invariant Feature Transform(SIFT) algorithm
        :param img: float/int array, shape: (height, width, channel)
        :return sift_results (keypoints, descriptors)
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        # =========================================================================================================
        keypoints, descriptors = quick_calc(img)
        return keypoints, descriptors

    def vis(self, img):
        """
        Visualize the key points of the given image, you can save the result as an image or just plot it.
        :param img: float/int array, shape: (height, width, channel)
        :return your own stuff (DIY is ok)
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        # =========================================================================================================
        keypoints, _ = quick_calc(img)
        random.shuffle(keypoints)
        num = 0
        while num < len(keypoints):
            new_img = np.copy(img)
            for i in range(200):
                draw_a_keypoint(new_img, keypoints[num])
                num += 1
                if num >= len(keypoints):
                    break
            cv2.imwrite('vis'+ str(num // 100) +'.jpg', new_img)
        return new_img

    def match(self, img1, img2):
        """
        Match keypoints between img1 and img2 and draw lines between the corresponding keypoints;
        you can save the result as an image or just plot it.
        :param img1: float/int array, shape: (height, width, channel)
        :param img1: float/int array, shape: (height, width, channel)
        :return your own stuff (DIY is ok)
        """ 
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        # =========================================================================================================
        
        kp1, des1 = quick_calc(img1)
        kp2, des2 = quick_calc(img2)
        print(len(kp1), len(kp2))
        new1 = np.copy(img1)
        new2 = np.copy(img2)
        for key in kp1:
            draw_a_spot(new1, key)
        for key in kp2:
            draw_a_spot(new2, key)

        matches = []

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=1)

        height = max(img1.shape[0], img2.shape[0])
        width = img1.shape[1] + img2.shape[1] 
        new_image = np.zeros([height, width, 3])
        padding1 = (height - img1.shape[0]) // 2
        new_image[padding1: img1.shape[0] + padding1, 0 : img1.shape[1], :] = new1
        padding2 = (height - img2.shape[0]) // 2
        new_image[padding2: img2.shape[0] + padding2, img1.shape[1] : width, :] = new2
        cv2.imwrite('match_vis.jpg', new_image)
        for m in matches:
            key1 = kp1[m[0].queryIdx]
            key2 = kp2[m[0].trainIdx]
            pt1 = (int(key1.pt[0]), int(key1.pt[1] + padding1))
            pt2 = (int(key2.pt[0] + img1.shape[1]), int(key2.pt[1] + padding2))
            cv2.line(new_image, pt1, pt2, (255, 0, 0))
        return new_image


if __name__ == '__main__':
    # img_path = 'school_gate.jpeg'
    img_path = 'school_gate.jpeg'

    kwargs = {}

    # =========================================================================================================
    # for testing this demo:
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    img_trans = np.zeros([img.shape[1], img.shape[0], 3])
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            img_trans[i][j] = img[j][img.shape[1] - i - 1]
    sift = SIFT(**kwargs)
    vis = sift.vis(img)
    cv2.imwrite('vis.jpg', vis)
    # matching = sift.match(img, img_resize)
    # cv2.imwrite('match_resize.jpg', matching)
    # =========================================================================================================
