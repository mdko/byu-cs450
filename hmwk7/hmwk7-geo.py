import matplotlib.pyplot as pl
import numpy as np
import math
import matplotlib.cm as cm

#Image resizing (Magnification)
def parta():
	
	#Read in an image
	img = pl.imread('parrots.png')

	#Magnify it by a specified (integer) factor
	#using bilinear interpolation
	magnifications = (2, 3)

	for mag in magnifications:
		newimg = np.zeros(np.multiply(img.shape, mag), img.dtype)
		
		for r in range(newimg.shape[0]):
			for c in range(newimg.shape[1]):
				#Backward Mapping

				#Get inverse (where this current (r,c) pixel is 
				#located in the original image, which is probably not
				#exactly on a pixel)
				x, y = (r/float(mag), c/float(mag))
				fracx = x - int(x)
				fracy = y - int(y)

				#Get the four closest points (upperleft, upperright, bottomleft, bottomright)
				#from the original image to interpolate from
				p1 = (int(x), int(y))			#upperleft (closer to zero)
				p2 = (int(x), int(y) + 1)		#upperright
				p3 = (int(x) + 1, int(y))		#bottomleft
				p4 = (int(x) + 1, int(y) + 1)	#bottomright


				# Getting the original images surrounding values, checkign bounds
				old1 = img[p1]
				old2 = None
				if p2[1] == img.shape[1]:
					old2 = img[p1]
				else:
					old2 = img[p2]
				old3 = None
				if p3[0] == img.shape[0]:
					old3 = img[p1]
				else:
					old3 = img[p3]
				old4 = None
				if p4[0] == img.shape[0] or p4[1] == img.shape[1]:
					old4 = img[p1]
				else:
					old4 = img[p4]

				#Interpolate (using the unit-square approach from wikipedia and www.cse.unr.edu/~looney/cs674/mgx6/unit6.pdf)
				val = old1*(1-fracx)*(1-fracy) + old3*fracx*(1-fracy) + old2*(1-fracx)*fracy + old4*fracx*fracy
				newimg[r,c] = val

		pl.imsave("ParrotsMagnifiedBy%d"%(mag),newimg, cmap=cm.Greys_r)

# Image Resizing (Reduction)
def partb():
	images = ('parrots.png', 'testImage2.png', 'testImage3.png', 'testImage4.png')
	
	for imgname in images:
		#Read in an image
		img = pl.imread(imgname)

		#Reduce it by a specified (integer) factor
		reductions = (4, 8)

		for red in reductions:
			newimg = np.zeros(np.multiply(img.shape, 1/float(red)), img.dtype)
			
			for r in range(newimg.shape[0]):
				for c in range(newimg.shape[1]):

					val = get_average((r*red,c*red), red, img)
					newimg[r,c] = val

			pl.imsave("%sReducedBy%d.png"%(imgname, red), newimg, cmap=cm.Greys_r)


def get_average(point, reduction, img):
	#Get average from img in this layout:
	#[(r,c), ........, (r,c+red-1)]
	#[............................]
	#[(r+red-1),,(r+red-1,c+red-1)]

	mat = np.zeros(img.shape)
	mat[point[0]:point[0]+reduction, point[1]:point[1]+reduction] = np.ones((reduction,reduction))
	res = np.multiply(mat, img)
	res = res.sum()/float(reduction**2)
	return res


# Image Warping
def partc():
	images = ['parrots.png']

	#######################
	#1 120 degree increment
	#Read in an image
	img = pl.imread('parrots.png')

	#img.shape[0] --> height
	#img.shape[1] --> length
	origcorners = [(0, 0),
				   (0, img.shape[1]),
				   (img.shape[0], 0),
				   (img.shape[0], img.shape[1])]
	largestdist = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
	newimg = np.zeros((largestdist, largestdist))
	midpoint = np.divide(img.shape,2)
	angle = 120.0
	newcorners = []
	for corn in origcorners:
		x = corn[0]
		y = corn[1]
		newcorner = ((x-midpoint[0])*math.cos(math.radians(angle)) - (y-midpoint[1])*math.sin(math.radians(angle)) + midpoint[0],
					(x-midpoint[0])*math.sin(math.radians(angle)) + (y-midpoint[1])*math.cos(math.radians(angle)) + midpoint[1])
		newcorners.append(newcorner)
	#print newcorners

	xs = map(lambda(x): x[0], newcorners)
	ys = map(lambda(x): x[1], newcorners)
	minx = int(np.min(xs))
	maxx = int(np.max(xs))
	miny = int(np.min(ys))
	maxy = int(np.max(ys))

	for x in range(minx, maxx):
		for y in range(miny, maxy):
		 	u = (x-midpoint[0])*math.cos(math.radians(-angle)) - (y-midpoint[1])*math.sin(math.radians(-angle)) + midpoint[0]
			v = (x-midpoint[0])*math.sin(math.radians(-angle)) + (y-midpoint[1])*math.cos(math.radians(-angle)) + midpoint[1]
			
			#Get the four closest points
			#from the original image to interpolate from
			p1 = (int(u), int(v))			#upperleft (closer to zero)
			p2 = (int(u), int(v) + 1)		#upperright
			p3 = (int(u) + 1, int(v))		#bottomleft
			p4 = (int(u) + 1, int(v) + 1)	#bottomright
			
			val = 0.0
			if u < 0 or u >= img.shape[0] - 1 or v < 0 or v >= img.shape[1] - 1:
				val = 0.0
			else:				
				# Getting the original images surrounding values
				old1 = img[p1]
				old2 = img[p2]
				old3 = img[p3]
				old4 = img[p4]
				
				#Interpolate
				b = np.array([old1, old2, old3, old4])
				a = np.array([[p1[0], p1[1], p1[0]*p1[1], 1], [p2[0], p2[1], p2[0]*p2[1], 1],
							  [p3[0], p3[1], p3[0]*p3[1], 1], [p4[0], p4[1], p4[0]*p4[1], 1]])
				coef = np.linalg.solve(a, b)
				val = coef.dot(np.array([u, v, u*v, 1]))

			r = x + (0 - minx)
			c = y + (0 - miny)
			newimg[r,c] = val

	pl.imsave("ParrotsRotatedOnly120.png", newimg, cmap=cm.Greys_r)


	#######################
	#8 15 degree increments

	img = pl.imread('parrots.png')
	for n in range(8):
		print n
		#img.shape[0] --> height
		#img.shape[1] --> length
		origcorners = [(0, 0),
					   (0, img.shape[1]),
					   (img.shape[0], 0),
					   (img.shape[0], img.shape[1])]
		largestdist = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
		newimg = np.zeros((largestdist, largestdist))
		midpoint = np.divide(img.shape,2)
		angle = 15.0
		newcorners = []
		for corn in origcorners:
			x = corn[0]
			y = corn[1]
			newcorner = ((x-midpoint[0])*math.cos(math.radians(angle)) - (y-midpoint[1])*math.sin(math.radians(angle)) + midpoint[0],
						(x-midpoint[0])*math.sin(math.radians(angle)) + (y-midpoint[1])*math.cos(math.radians(angle)) + midpoint[1])
			newcorners.append(newcorner)
		#print newcorners

		xs = map(lambda(x): x[0], newcorners)
		ys = map(lambda(x): x[1], newcorners)
		minx = int(np.min(xs))
		maxx = int(np.max(xs))
		miny = int(np.min(ys))
		maxy = int(np.max(ys))

		for x in range(minx, maxx):
			for y in range(miny, maxy):
			 	u = (x-midpoint[0])*math.cos(math.radians(-angle)) - (y-midpoint[1])*math.sin(math.radians(-angle)) + midpoint[0]
				v = (x-midpoint[0])*math.sin(math.radians(-angle)) + (y-midpoint[1])*math.cos(math.radians(-angle)) + midpoint[1]
				
				#Get the four closest points
				#from the original image to interpolate from
				p1 = (int(u), int(v))			#upperleft (closer to zero)
				p2 = (int(u), int(v) + 1)		#upperright
				p3 = (int(u) + 1, int(v))		#bottomleft
				p4 = (int(u) + 1, int(v) + 1)	#bottomright
				
				val = 0.0
				if u < 0 or u >= img.shape[0] - 1 or v < 0 or v >= img.shape[1] - 1:
					val = 0.0
				else:				
					# Getting the original images surrounding values
					old1 = img[p1]
					old2 = img[p2]
					old3 = img[p3]
					old4 = img[p4]
					
					#Interpolate
					b = np.array([old1, old2, old3, old4])
					a = np.array([[p1[0], p1[1], p1[0]*p1[1], 1], [p2[0], p2[1], p2[0]*p2[1], 1],
								  [p3[0], p3[1], p3[0]*p3[1], 1], [p4[0], p4[1], p4[0]*p4[1], 1]])
					coef = np.linalg.solve(a, b)
					val = coef.dot(np.array([u, v, u*v, 1]))

				r = x + (0 - minx)
				c = y + (0 - miny)
				newimg[r,c] = val
		img = newimg

		#pl.imshow(img, cmap=cm.Greys_r)
		pl.show()
		pl.imsave("ParrotsRotated%d.png"%((n+1)*15), img, cmap=cm.Greys_r)

	
def run():
	parta()
	partb()
	partc()


if __name__ == "__main__":
	run()