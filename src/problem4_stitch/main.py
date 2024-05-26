import cv2
import numpy as np
from sift import SIFT, quick_calc
from ransac import ransac


class Stitch(SIFT):
    def __init__(self, mid, **kwargs):
        """
        Implement panorama stitching with RANSAC
        :param kwargs: other hyperparameters of SIFT, such as sigma, blur ratio, border, etc.
        """
        super().__init__(**kwargs)
        self.mid = mid
        pass

    # =========================================================================================================
    # TODO: you can add other functions here or files in this directory
    # =========================================================================================================

    def stitch_img_lst(self, img_lst):
        """
        Stitch a list of image in order (NOT RANDOM),
        you can save the result as an image or just plot it.
        :param img_lst: a list of images, the shape of image is (height, width, channel)
        :return your own stuff (DIY is ok)
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        # =========================================================================================================

        kp = []
        des = []
        size = len(img_lst)
        mid = self.mid
        for i in range(size):
            kp_, des_ = quick_calc(img_lst[i], self.sigma, self.num_intervals, self.border)
            kp.append(kp_)
            des.append(des_)
            print(len(kp[i]))
        h, w, _ = img_lst[0].shape
        M = []
        for i in range(size - 1):
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des[i], des[i + 1], k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            src_pts = np.float32([kp[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[i + 1][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            for i in range(src_pts.shape[0]):
                src_pts[i][0] += (mid * w, h)
                dst_pts[i][0] += (mid * w, h)

            _, m1 = ransac(dst_pts, src_pts)
            _, m2 = ransac(src_pts, dst_pts)
            M.append([m1, m2])

        new_img = np.zeros([3 * h, (size+1) * w, 3])
        new_img[h : 2 * h, mid * w : mid * w + w, :] = img_lst[mid]
        m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for j in range(mid - 1, -1, -1):
            m = np.matmul(M[j][1], m)
            new = np.zeros([3 * h, (size+1) * w, 3])
            new[h : 2 * h, mid * w : mid * w + w, :] = img_lst[j]
            new = cv2.warpPerspective(new, m, (new_img.shape[1], new_img.shape[0]))
            for x in range(3 * h):
                for y in range((size+1) * w):
                    if (new_img[x][y] == 0).all():
                        new_img[x][y] += new[x][y]
        m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for j in range(mid, len(img_lst) - 1):
            m = np.matmul(M[j][0], m)
            new = np.zeros([3 * h, (size+1) * w, 3])
            new[h : 2 * h, mid * w : mid * w + w, :] = img_lst[j + 1]
            new = cv2.warpPerspective(new, m, (new_img.shape[1], new_img.shape[0]))
            for x in range(3 * h):
                for y in range((size+1) * w):
                    if (new_img[x][y] == 0).all():
                        new_img[x][y] += new[x][y]
        cv2.imwrite('stich.png', new_img)
        return new_img


if __name__ == '__main__':
    img1 = cv2.imread('building_image/building_1.jpg')
    img2 = cv2.imread('building_image/building_2.jpg')
    img3 = cv2.imread('building_image/building_3.jpg')
    img4 = cv2.imread('building_image/building_4.jpg')
    img_lst = [img1,img2, img3, img4]

    # =========================================================================================================
    # for testing this demo:
    stitch = Stitch()
    stitch.stitch_img_lst(img_lst, 1)
    # =========================================================================================================
