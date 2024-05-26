import cv2
from matplotlib import pyplot as plt
import numpy as np
from sift.main import SIFT, quick_calc, draw_a_keypoint

def ransac(src, dst, num_ite = 30, threshold = 5, worst = 10):
    inner = []
    max_M = 0
    size = dst.shape[0]
    for ite in range(num_ite):
        new_inner = []
        random = np.random.randint(0, size-1, 4)
        M = cv2.findHomography(src[random], dst[random], 0)[0]
        new = cv2.perspectiveTransform(src, M)
        for i in range(size):
            norm = np.linalg.norm(new[i]- dst[i])
            if norm < threshold:
                new_inner.append(i)
        if len(new_inner) > len(inner):
            inner = new_inner
            max_M = M
        print(len(new_inner), len(inner))
    if len(inner) < worst:
        return None
    return np.array(inner), max_M


class R_SIFT(SIFT):
    def __init__(self, **kwargs):
        """
        Implement Scale-Invariant Feature Transform(SIFT) algorithm with RANSAC
        :param kwargs: other hyperparameters, such as sigma, blur ratio, border, etc.
        """
        super().__init__(**kwargs)
        pass

    # =========================================================================================================
    # TODO: you can add other functions here or files in this directory
    # =========================================================================================================

    def match(self, img1, img2):
        """
        Match keypoints between img1 and img2 and draw lines between the corresponding keypoints with RANSAC,
        you can save the result as an image or just plot it.
        :param img1: float/int array, shape: (height, width, channel)
        :param img1: float/int array, shape: (height, width, channel)
        :return your own stuff (DIY is ok)
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        # =========================================================================================================
        MIN_MATCH_COUNT = 10

        # Compute SIFT keypoints and descriptors
        kp1, des1 = quick_calc(img1, self.sigma, self.num_intervals, self.border)
        kp2, des2 = quick_calc(img2, self.sigma, self.num_intervals, self.border)
        print(len(kp1), len(kp2))
        new1 = np.copy(img1)
        new2 = np.copy(img2)
        for key in kp1:
            draw_a_keypoint(new1, key)
        for key in kp2:
            draw_a_keypoint(new2, key)

        # Initialize and use FLANN
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            inner = []
            ite = 0
            while True:
                src_pts = np.float32([item for index, item in enumerate(src_pts) if index not in inner])
                dst_pts = np.float32([item for index, item in enumerate(dst_pts) if index not in inner])
                x = ransac(src_pts, dst_pts)
                if x == None:
                    break
                inner, M = x
                h, w, _ = img1.shape
                pts = np.float32([[0, 0],
                                [0, h - 1],
                                [w - 1, h - 1],
                                [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M) 
                new2 = cv2.polylines(new2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                print(len(inner))
                ransaced = []
                for i in inner:
                    ransaced.append([src_pts[i], dst_pts[i]])
                cv2.imwrite('vis'+str(ite)+'.png', new2)

                height = max(img1.shape[0], img2.shape[0])
                width = img1.shape[1] + img2.shape[1] 
                new_image = np.zeros([height, width, 3])
                padding1 = (height - img1.shape[0]) // 2
                new_image[padding1: img1.shape[0] + padding1, 0 : img1.shape[1], :] = new1
                padding2 = (height - img2.shape[0]) // 2
                new_image[padding2: img2.shape[0] + padding2, img1.shape[1] : width, :] = new2
                for m, n in ransaced:
                    pt1 = (int(m[0][0]), int(m[0][1] + padding1))
                    pt2 = (int(n[0][0] + img1.shape[1]), int(n[0][1] + padding2))
                    cv2.line(new_image, pt1, pt2, (255, 0, 0))
                cv2.imwrite('match'+str(ite)+'.jpg', new_image)
                ite += 1
            return
        else:
            print('no good enough match' + str(len(matches)))
            return None


if __name__ == '__main__':
    img1_path = 'book_reference.jpeg'
    img2_path = 'books.jpeg'

    kwargs = {}

    # =========================================================================================================
    # for testing this demo:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    r_sift = R_SIFT(**kwargs)
    r_sift.match(img1, img2)
    # =========================================================================================================
