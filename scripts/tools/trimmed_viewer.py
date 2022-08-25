import numpy as np
import p2t
import cv2

img = np.zeros((1000, 1000, 3))

shape = [
[40 , 0  ], 
[50 , 26 ],
[61 , 71 ],
[73 , 104],
[84 , 148],
[87 , 152],
[97 , 189],
[96 , 199],
[122, 199],
[123, 190],
[151, 185],
[170, 179],
[168, 165],
[166, 164],
[161, 138],
[147, 90 ],
[149, 76 ],
[157, 77 ],
[157, 86 ],
[160, 87 ],
[162, 103],
[170, 123],
[182, 174],
[186, 175],
[187, 167],
[193, 167],
[195, 172],
[199, 172],
[199, 62 ],
[196, 62 ],
[184, 13 ],
[173, 19 ],
[165, 19 ],
[158, 0  ]
]
holes = [
[
[182, 148], 
[189, 149],
[192, 160],
#[191, 167],
[184, 166],
[181, 155]
],

[
[174, 117], 
[181, 118],
[185, 140],
[187, 141],
[186, 148],
[179, 147],
[173, 126]
],

[
[163, 75], 
[170, 76],
[173, 89],
[172, 96],
[165, 95],
[162, 82]
],

[
[158, 56],
[165, 57],
[165, 68],
[158, 67]
],

[
[152, 38],
[159, 37],
[160, 44],
[153, 45]
]
]

tmp = []
for i in range(len(shape)):
    tmp.append(p2t.Point(shape[i][0], shape[i][1]))
cdt = p2t.CDT(tmp)

for hole in holes:
    tmp_hole = []
    for ph in hole:
        tmp_hole.append(p2t.Point(ph[0], ph[1]))
    cdt.add_hole(tmp_hole)

tri = cdt.triangulate()

for i in range(len(shape)):
    cv2.line(img, (shape[i-1][0]*5, shape[i-1][1]*5), (shape[i][0]*5, shape[i][1]*5), (255, 255, 255), thickness=1)
for hole in holes:
    for i in range(len(hole)):
        cv2.line(img, (hole[i-1][0]*5, hole[i-1][1]*5), (hole[i][0]*5, hole[i][1]*5), (255, 255, 0), thickness=1)

while True:
    for i in range(len(shape)):
        tmp_img = img.copy()
        cv2.circle(tmp_img, (shape[i][0]*5, shape[i][1]*5), 2, (255, 0, 0), -1)
        #tmp_img = cv2.resize(tmp_img, (tmp_img.shape[0] * 5, tmp_img.shape[1] * 5))
        cv2.imshow("test", tmp_img)
        cv2.waitKey(500)
    for hole in holes:
        for i in range(len(hole)):
            tmp_img = img.copy()
            cv2.circle(tmp_img, (hole[i][0]*5, hole[i][1]*5), 2, (0, 0, 255), -1)
            #tmp_img = cv2.resize(tmp_img, (tmp_img.shape[0] * 5, tmp_img.shape[1] * 5))
            cv2.imshow("test", tmp_img)
            cv2.waitKey(500)
