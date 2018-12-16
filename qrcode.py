import numpy as np
import cv2

# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    A = np.array([[u[0][0], u[0][1], 1, 0, 0, 0, -1 * u[0][0] * v[0][0], -1 * u[0][1] * v[0][0]],
                  [0, 0, 0, u[0][0], u[0][1], 1, -1 * u[0][0] * v[0][1], -1 * u[0][1] * v[0][1]],
                  [u[1][0], u[1][1], 1, 0, 0, 0, -1 * u[1][0] * v[1][0], -1 * u[1][1] * v[1][0]],
                  [0, 0, 0, u[1][0], u[1][1], 1, -1 * u[1][0] * v[1][1], -1 * u[1][1] * v[1][1]],
                  [u[2][0], u[2][1], 1, 0, 0, 0, -1 * u[2][0] * v[2][0], -1 * u[2][1] * v[2][0]],
                  [0, 0, 0, u[2][0], u[2][1], 1, -1 * u[2][0] * v[2][1], -1 * u[2][1] * v[2][1]],
                  [u[3][0], u[3][1], 1, 0, 0, 0, -1 * u[3][0] * v[3][0], -1 * u[3][1] * v[3][0]],
                  [0, 0, 0, u[3][0], u[3][1], 1, -1 * u[3][0] * v[3][1], -1 * u[3][1] * v[3][1]]
                ])

    b = np.array([[v[0][0]],
                  [v[0][1]],
                  [v[1][0]],
                  [v[1][1]],
                  [v[2][0]],
                  [v[2][1]],
                  [v[3][0]],
                  [v[3][1]]
                ])

    tmp = np.dot(np.linalg.inv(A), b)
    H = np.array([[tmp[0][0], tmp[1][0], tmp[2][0]],
                  [tmp[3][0], tmp[4][0], tmp[5][0]],
                  [tmp[6][0], tmp[7][0], 1]
                 ])

    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    orig_corner = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    matrix = solve_homography(orig_corner, corners)

    # apply matrix to every point
    for i in range(h):
        for j in range(w):
            tmp = np.dot(matrix, np.array([[j, i, 1]]).T)
            x, y = int(tmp[0][0] / tmp[2][0]), int(tmp[1][0] / tmp[2][0])
            canvas[y][x] = img[i][j]
    return canvas


def interpolation(img, new_x, new_y):
    fx = round(new_x - int(new_x), 2)
    fy = round(new_y - int(new_y), 2)

    p = np.zeros((3,))
    p += (1 - fx) * (1 - fy) * img[int(new_y), int(new_x)]
    p += (1 - fx) * fy * img[int(new_y) + 1, int(new_x)]
    p += fx * (1 - fy) * img[int(new_y), int(new_x) + 1]
    p += fx * fy * img[int(new_y) + 1, int(new_x) + 1]

    return p


def backward_warpping(img, output, corners):
    h, w, ch = output.shape
    img_corner = np.array([[0, 0], [w-1, 0], 
                           [0, h-1], [w-1, h-1]
                          ])
    homography_matrix = solve_homography(img_corner, corners)
    for y in range(h):
        for x in range(w):
            new_pos = np.dot(homography_matrix, np.array([[x, y, 1]]).T)
            new_x, new_y = new_pos[0][0] / new_pos[2][0], new_pos[1][0] / new_pos[2][0]
            res = interpolation(img, new_x, new_y)
            output[y][x] = res

    return output


def main():
    
    # recover/unwarp qrcode from screen 
    img = cv2.imread('./input/screen.jpg')
    
    output = np.zeros((300, 300, 3))
    
    # corner location
    screen_corner = np.array([[1038, 367], [1102, 393], [981, 558], [1035, 601]])
    output = backward_warpping(img, output, screen_corner)
    
    cv2.imwrite('recover.png', output)

if __name__ == '__main__':
    main()
