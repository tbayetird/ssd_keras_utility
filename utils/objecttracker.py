# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
from .objects import Object
import numpy as np


def treatCentroids(inputCentroids):
	##TODO : change this to use Non maximum suppression ! (en fonction des labels)
	D = dist.cdist(np.array(inputCentroids), inputCentroids)
	toDel=[]
	if len(D)<2:
		return inputCentroids
	for (i,rows) in enumerate(D) :
		if i in toDel:
			continue
		for (i,col) in enumerate(rows):
			if(col<50 and col>0): #TODO : val en param
				toDel.append(i)
	outputCentroids = np.zeros((len(inputCentroids)-len(toDel),
	 									2), dtype="int")
	deletedCount=0
	for (i,rows) in enumerate(D):
		if i in toDel:
			deletedCount+=1
			continue
		outputCentroids[i-deletedCount]=inputCentroids[i]
	return outputCentroids

class ObjectTracker():

	def __init__(self, maxDisappeared=20,maxDistance=200):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.count=0

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared
		self.maxDistance = maxDistance


	def register(self, rect, centroid,maxDistance):
		# print ("[INFO] -- objet enregistre avec l'ID : {}".format(self.nextObjectID))
		self.objects[self.nextObjectID] = Object([self.nextObjectID,centroid,rect,maxDistance])
		self.nextObjectID += 1
		self.count +=1


	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]

	def getTotalDetectedCentroids(self):
		return self.count

	def updateTrackedObjects(self,rects,inputCentroids):
		# grab the set of object IDs and corresponding centroids
		objectIDs = list(self.objects.keys())
		objectCentroids = list()
		for ID in objectIDs:
			objectCentroids.append(self.objects[ID].getTempPredictedCentroid())

		# compute the distance between each pair of object
		# centroids and input centroids
		D = dist.cdist(np.array(objectCentroids), inputCentroids)
		#  (1) find the smallest value in each row and then
		# (2) sort the row indexes based on their minimum values
		rows = D.min(axis=1).argsort()
		# finding the smallest value in each column and then
		# sorting using the previously computed row index list
		cols = D.argmin(axis=1)[rows]
		usedRows = set()
		usedCols = set()
		# print("[INFO] D : {}".format(D))
		# print("[INFO] : rows : {}".format(rows))
		# print("[INFO] : cols : {}".format(cols))
		for (row, col) in zip(rows, cols):
			objectID = objectIDs[row]
			object=self.objects[objectID]
			(x,y)=inputCentroids[col]
			distance = D[row][col]
			if row in usedRows or col in usedCols:
				# print("Objet {} refuse car donnee deja utilisee".format(objectID))
				# print("D : {}".format(D))
				# print("rows {} ; row : {}".format(rows,row))
				# print("cols {} ;  col : {}".format(cols,col))
				continue
			if distance>object.getMaxDistance():
				# print("Objet {} reconnu comme étant trop loin !".format(objectID))
				# print("distance superieure a {}".format(object.getMaxDistance()))
				continue

			if(object.checkDirection(x,y) or distance < 30):
				#TODO : tout mettre dans une updateObject
				object.update(np.array([x,y]),rects[col])
				# object.setCentroid(inputCentroids[col])
				# object.setRect(rects[col])
				# object.setDisappeared(0)

				usedRows.add(row)
				usedCols.add(col)

		# compute both the row and column index we have NOT yet
		# examined
		unusedRows = set(range(0, D.shape[0])).difference(usedRows)
		unusedCols = set(range(0, D.shape[1])).difference(usedCols)

		for row in unusedRows:
			# object seems to have disappeared
			objectID = objectIDs[row]
			self.objects[objectID].addDisappeared()
			if self.objects[objectID].getDisapeared() > self.maxDisappeared:
				self.deregister(objectID)

		for col in unusedCols:
			self.register(rects[col],inputCentroids[col],self.maxDistance)

		# return the set of trackable objects
		return self.objects

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			toDeregister=[]
			for objectID in self.objects.keys():
				self.objects[objectID].addDisappeared()

				if self.objects[objectID].getDisapeared() > self.maxDisappeared:
					toDeregister.append(objectID)
			# return early as there are no centroids or tracking info
			# to update
			for i in toDeregister :
				self.deregister(i)
			return self.objects
		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
		#First of all, sometimes objects can be detected as multple classes
		#We need to treat the detected objects first
		inputCentroids=treatCentroids(inputCentroids)
		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			# print("[INFO] -- creating objects because of lack of already tracked ones")
			for i in range(0, len(inputCentroids)):
				self.register(rects[i],inputCentroids[i],self.maxDistance)
			return self.objects
		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			return self.updateTrackedObjects(rects,inputCentroids)
