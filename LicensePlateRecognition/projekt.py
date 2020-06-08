import cv2
import numpy as np
import imutils

images = ['nni.png', 'nol.png', 'rze.png', 'wt.png', 'wu.png']

for im in images:
	img = cv2.imread(im)

	gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	re_gray = imutils.resize(gr, height=500)
	cv2.imshow("gray", re_gray)

	height = gr.shape[0]
	width = gr.shape[1]

	i = 0
	while i < height/2:
		j = 0
		while j < width:
			gr[i, j] = 0
			j += 1
		i += 1

	re_mask = imutils.resize(gr, height=500)
	cv2.imshow("masked", re_mask)

	_, tresh = cv2.threshold(gr, 125, 255, cv2.THRESH_BINARY_INV)
	re_tresh = imutils.resize(tresh, height=500)
	cv2.imshow("tresh", re_tresh)

	kernel = np.ones((3,3), np.uint8)
	tresh = cv2.dilate(tresh, kernel, iterations = 3)
	tresh = cv2.erode(tresh, kernel, iterations = 2)
	re_morph = imutils.resize(tresh, height=500)
	cv2.imshow("morph", re_morph)

	edged = cv2.Canny(tresh, 50, 200)
	re_edged = imutils.resize(edged, height=500)
	cv2.imshow("edged", re_edged)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:500]

	image = img.copy()
	cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)

	screenCnt = None

	for c in cnts:
	    peri = cv2.arcLength(c, True)
	    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
	    x,y,w,h = cv2.boundingRect(c)
	    surface = w*h
	    if len(approx) == 4 and surface > 200000:
	    	screenCnt = approx

	mask = np.zeros(gr.shape,np.uint8)
	new_image = cv2.drawContours(mask,[screenCnt],0,255,-1)
	new_image = cv2.bitwise_and(image,image,mask=mask)

	(x, y) = np.where(mask == 255)
	(topx, topy) = (np.min(x), np.min(y))
	(bottomx, bottomy) = (np.max(x), np.max(y))
	cropped = gr[topx:bottomx+10, topy:bottomy+10]

	cv2.imshow("cropped", cropped)

	_, gray = cv2.threshold(cropped, 40, 255, cv2.THRESH_BINARY_INV)

	kernel = np.ones((3,3), np.uint8)
	gray = cv2.dilate(gray, kernel, iterations = 4)
	gray = cv2.erode(gray, kernel, iterations = 3)

	cv2.imshow("tresh", gray)

	templates = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', '0.png', 'A.png', 'B.png', 'C.png', 'D.png', 'E.png', 'F.png', 'G.png', 'H.png',
	    'I.png', 'J.png', 'K.png', 'L.png', 'M.png', 'N.png', 'O.png', 'P.png', 'R.png', 'S.png', 'T.png', 'U.png', 'W.png', 'X.png', 'Y.png', 'Z.png']

	tempList = []

	for temp in templates:
	    template = cv2.imread(temp)
	    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	    (tH, tW) = template.shape[:2]

	    found = None
	    bufforList = []

	    for scale in np.linspace(0.2, 0.8, 20)[::-1]:
	        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
	        r = gray.shape[1] / float(resized.shape[1])
	        if resized.shape[0] < tH or resized.shape[1] < tW:
	            break
	        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
	        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

	        if (temp[0] == 'I'):
	            val = 0.9
	        else:
	            val = 0.75
	        
	        if (maxVal > val):
	            found = (maxVal, maxLoc, r)
	            check = 1
	            for bl in bufforList:
	                (_, mL, _) = bl
	                if (abs(maxLoc[0] - mL[0]) < 60):
	                    check = 0
	            if (check == 1):
	                char = (temp[0], (maxLoc[0]*r, maxLoc[1]*r), r)
	                tempList.append(char)
	            bufforList.append(found)

	sortedList = sorted(tempList, key=lambda char: char[1][0])

	plateNumber = ''

	for s in sortedList:
	    plateNumber = plateNumber + s[0]

	print(plateNumber)

	cv2.putText(
	    img,
	    "Numer rejestracyjny: " + str(plateNumber),
	    (100, 100),
	    cv2.FONT_HERSHEY_COMPLEX,
	    4,
	    (0, 0, 0),
	    10,
	)

	scale_percent = 20
	width = int(img.shape[1]*scale_percent/100)
	height = int(img.shape[0]*scale_percent/100)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	cv2.imshow("result", resized)
	cv2.waitKey(0)

cv2.destroyAllWindows()