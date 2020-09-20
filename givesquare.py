import cv2 as cv
import numpy as np

def findCorners(image):
    output = [];
    greyimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    colorimg = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    ret, thresh = cv.threshold(greyimg,150,255,cv.THRESH_BINARY)
    cnts, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts.sort(key=lambda c: -cv.contourArea(c))
    cnts = [c for c in cnts if cv.contourArea(c) > 100 and cv.contourArea(c) < 220000]
    cv.drawContours(img, cnts, -1, (0, 255, 0), 3)
    red_lower = np.array([136, 87, 111], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv.inRange(colorimg, red_lower, red_upper)
    black_lower = np.array([0, 0, 0], np.uint8) 
    black_upper = np.array([180, 255, 10], np.uint8) 
    black_mask = cv.inRange(colorimg, black_lower, black_upper) 
    blue_lower = np.array([94, 80, 2], np.uint8) 
    blue_upper = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv.inRange(colorimg, blue_lower, blue_upper)
    kernal = np.ones((5, 5), "uint8") 
    red_mask = cv.dilate(red_mask, kernal) 
    res_red = cv.bitwise_and(img, img,  mask = red_mask) 
    black_mask = cv.dilate(black_mask, kernal) 
    res_black = cv.bitwise_and(img, img, mask = black_mask) 
    blue_mask = cv.dilate(blue_mask, kernal) 
    res_blue = cv.bitwise_and(img, img, mask = blue_mask)
    contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    red = [1 if cv.matchShapes(redcnt,cnt,1,0.0) < 0.05 else 0 for redcnt, cnt in zip(contours, cnts)]
    contours, hierarchy = cv.findContours(blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    blue = [1 if cv.matchShapes(redcnt,cnt,1,0.0) < 0.05 else 0 for redcnt, cnt in zip(contours, cnts)]
    contours, hierarchy = cv.findContours(black_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    black = [1 if cv.matchShapes(redcnt,cnt,1,0.0) < 0.05 else 0 for redcnt, cnt in zip(contours, cnts)]
    maxtop = 0
    maxright = 0
    maxleft = 0;
    maxbottom = 0;
    for cnt, red, blue, black in zip(cnts, red, blue, black):
        left = tuple(cnt[cnt[:,:,0].argmin()][0])
        if left[0] <= maxleft:
            maxleft = left[0]
        right = tuple(cnt[cnt[:,:,0].argmax()][0])
        if right[0] >= maxright:
            maxright = right[0]
        top = tuple(cnt[cnt[:,:,1].argmin()][0])
        if top[1] >= maxtop:
            maxtop = top[1]
        bottom = tuple(cnt[cnt[:,:,1].argmax()][0])
        if bottom[1] <= maxbottom:
            maxbottom = bottom[0]
        color = red * 1 + blue * 2 + black * 4
        output.append({'type': color, 'points':[(left[0], bottom[1]), (left[0], top[1]), (right[0], top[1]), (right[0], bottom[1])]})
    for out in output:
        tmp = out['points']
        out['points'] = [((tmp[0][0]) / (100) , (tmp[0][1]) / (100)), ((tmp[1][0]) / (100) , (tmp[1][1]) / (100)), ((tmp[2][0]) / (100) , (tmp[2][1]) / (100)), ((tmp[3][0]) / (100) , (tmp[3][1]) / (100))]
    return output

if __name__ == '__main__':
    img = cv.imread('test.jpg')
    output, img, greyimg = findCorners(img)
    cv.imwrite('test.png', img)
    cv.imwrite('testgrey.png', greyimg)
    print(output)