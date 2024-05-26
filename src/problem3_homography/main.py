import cv2
import numpy as np


# =========================================================================================================
# TODO: you can add other functions here or files in this directory
# =========================================================================================================

def calculate_distance(img):
    """
    Calculate the required distance through homography (metric: meter).
    Error less than 5% is ok.
    :param img: float/int array, shape: (height, width, channel)
    :return (distance 1, distance 2, distance 3), type: float
    """
    # =========================================================================================================
    # TODO: Please fill this part with your code
    # But DO NOT change this interface
    referee = [380, 20, 1]
    football = [362, 256, 1]
    leftfoot = [300, 784, 1]
    goalpost = [320, 880, 1]
    mat1 = np.array([[275, 656, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 275, 656, 1, 0, 0],
            [339, 995, 1, 0, 0, 0, -339 * 1832, -995 * 1832], 
            [0, 0, 0, 339, 995, 1, 0, 0],
            [352, 858, 1, 0, 0, 0, -352 * 1832, -858 * 1832], 
            [0, 0, 0, 352, 858, 1, -352 * 550, -858 * 550],
            [283, 543, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 283, 543, 1, -283 * 550, -543 * 550]])
    mat2 = np.array([0, 0, 1832, 0, 1832, 550, 0, 550])
    solve = np.linalg.solve(mat1, mat2)
    H = np.zeros(9)
    H[:-1] = solve
    H[-1] = 1
    H = H.reshape((3,3))
    referee = np.matmul(H, referee)
    football = np.matmul(H, football)
    leftfoot = np.matmul(H, leftfoot)
    goalpost = np.matmul(H, goalpost)
    distance_2 = np.round(abs(football[1])) / 100
    distance_1 = np.round(np.linalg.norm(football - referee)) / 100
    distance_3 = np.round(abs(leftfoot[0] - goalpost[0])) / 100
    print(distance_1, distance_2, distance_3)
    # =========================================================================================================

    return distance_1, distance_2, distance_3


if __name__ == '__main__':
    img_path = 'football.png'

    img = cv2.imread(img_path)

    # =========================================================================================================
    # for testing this demo:
    distance_1, distance_2, distance_3 = calculate_distance(img)
    # =========================================================================================================
