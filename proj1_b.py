import cv2
import numpy as np
import sys
import math

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]


if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)
#convert w1.h1,w2,h2
rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))
uw = 4 * 0.95 / (0.95 + 15 + 3 * 1.09)
vw = 9 / (0.95 + 15 + 3 * 1.09)
L_MAX = 0
L_MIN = 100
helper = []
for i in range(0,101):
    helper.append(0)

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

tmp = np.zeros([rows, cols, bands], dtype=np.int16)

def invgamma( color):
    if (color<0.03928):
        return color/12.92
    else:
        return math.pow((color+0.055)/1.055,2.4)

def gamma (color):
    if(color < 0.00304):
        return 12.92 * color
    else:
        return 1.055 * math.pow(color, 1/2.4) - 0.055

# bgr to luv

for i in range(0, round(rows-1)) :
    for j in range(0, round(cols-1)) :
        b, g, r = inputImage[i, j]
        if(b==0 and g==0 and r==0):
            tmp[i,j]=[0,0,0]
            break
        b ,g, r = invgamma(b/255),invgamma(g/255), invgamma(r/255)
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b
        y = 0.212671 * r + 0.71516 * g + 0.072109 * b
        x = 0.412453 * r + 0.35758 * g + 0.180423 * b
        t = y
        if (t > 0.008856):
            L = 116 * math.pow(t,1/3) - 16
        else:
            L = 903.3 * t
        if (( i>=H1 and i<=H2 ) and ( j>=W1 and j<=W2 )):
            L_MAX = max(L_MAX, L)
            L_MIN = min(L_MIN, L)
        d = x + 15 * y + 3 * z
        u1 = 4 * x / d
        v1 = 9 * y / d
        # if(i==H1 and j==W1):
        #      print("u1,v1:",u1,v1)
        u = 13 * L * (u1 - uw)
        v = 13 * L * (v1 - vw)
        tmp[i, j] = [L, u, v]
        L = tmp[i, j][0]
        helper[L]+=1
for i in range(1,101):
    helper[i]+=helper[i-1]
    

outputImage = np.copy(inputImage)

# luv to bgr
for i in range(0, round(rows-1)) :
    for j in range(0, round(cols-1)) :
        L, u, v = tmp[i, j]
        if(L < L_MIN):
            L = 0
        elif (L > L_MAX):
            L = 100
        else:
            L = math.floor(101 * (helper[L-1] + helper[L]) / (2 * helper[100]))
        if(L != 0):
            u1 = (u + 13 * uw * L )/ (13 * L)
            v1 = (v + 13 * vw * L )/ (13 * L)
        else:
            u1, v1 = 0, 0
        if (L > 7.9996):
            y = math.pow( (L + 16) / 116, 3)
        else:
            y = L / 903.3
        if(v1 == 0):
            x, z = 0, 0
        else:
            x = y * 2.25 * u1 / v1
            z = y * (3 - 0.75 * u1 - 5 * v1) / v1

        r = gamma(3.240479 * x - 1.53715 * y - 0.498535 * z ) * 255
        g = gamma(-0.969256 * x + 1.875991 * y + 0.041556* z) * 255
        b = gamma(0.055648 * x - 0.204043 *y + 1.057311 * z) * 255
        if( r < 0):
            r = 0
        elif(r > 255):
            r = 255
        else:
            r = r
        if( g < 0):
            g = 0
        elif(g > 255):
            g = 255
        else:
            g = g
        if( b < 0):
            b = 0
        elif(b > 255):
            b = 255
        else:
            b = b
        outputImage[i,j] = [b, g, r]
        if (i==H1 and j==W1):
            print('bgr',b,g,r)
cv2.imshow("output:", outputImage)
cv2.imwrite(name_output, outputImage);


# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
