import numpy as np


class Object():

	def __init__(self,args):
		# print("[INFO] -- creation d'un objet avec Id : {}".format(args[0]))
		self.objectID = args[0]
		#Motion model
		self.centroid = args[1]
		self.rect = args[2]
		self.maxDistance= args[3]
		self.disappeared = 0
		self.centroidBatch=[]
		self.rectBatch=[] #TODO : use these batchs to reduce rect trembling
		self.direction=np.array([0,0])
		self.speed=0
		self.debugText=""

		#Predicted model
		self.predictedCentroid=self.centroid
		self.tempPredictedCentroid=self.centroid


	### Get and sets
	def getID(self):
		return self.objectID

	def getCentroid(self):
		return self.centroid

	def setCentroid(self,centroid):
		self.centroid=centroid

	def setDisappeared(self,val):
		self.disappeared=val

	def getDisapeared(self):
		return self.disappeared

	def addDisappeared(self):
		self.disappeared +=1
		self.maxDistance = self.maxDistance + 30
		self.updateBatch(self.tempPredictedCentroid)
		self.predictNext()

	def getRect(self):
		return self.rect

	def setRect(self,rect):
		self.rect=rect

	def setDirection(self,val):
		self.direction=val

	def getDirection(self):
		return self.direction

	def getSpeed(self):
		return self.speed

	def setSpeed(self,val):
		self.speed=val

	def setPredictedCentroid(self,val):
		self.predictedCentroid=val

	def getPredictedCentroid(self):
		return self.predictedCentroid

	def setTempPredictedCentroid(self,val):
		self.tempPredictedCentroid=val

	def getTempPredictedCentroid(self):
		return self.tempPredictedCentroid

	def setMaxDistance(self,val):
		self.maxDistance=val

	def getMaxDistance(self):
		return self.maxDistance

	def getCentroidBatch(self):
		return self.centroidBatch

	### Update-specific methods
	def update(self,centroid,rect):
		self.setMaxDistance(200) # NOT GOOD - AVOID THIS 
		self.setDisappeared(0)
		self.updateCentroid(centroid)
		self.updateRect(rect)
		self.updateDirection()
		self.updateSpeed()
		self.predictNext()

	def updateRect(self,rect):
		self.setRect(rect)

	def updateBatch(self,elem):
		if len(self.centroidBatch)>4:
			self.centroidBatch.pop(0)
		self.centroidBatch.append(elem)

	def updateCentroid(self,centroid):
		self.updateBatch(centroid)
		weights = np.linspace(0.2,1,len(self.centroidBatch)).tolist()
		# weights.reverse()
		sum = 0
		for (i,w) in enumerate(weights) :
			#On décale le poids vers les hautes valeurs en retirant
			#une valeur constante
			weights[i]-=0.17
			sum +=w-0.17
		weights = list(i/sum for i in weights)
		# print("[INFO] normalized weights : {}".format(weights))
		newCentroid=np.array([0,0])
		for (i,centroid) in enumerate(self.centroidBatch):
			for k in range(2):
				newCentroid[k]+=weights[i]*centroid[k]
		# print("[INFO] centroid initial : {}".format(self.centroid))
		# print("[INFO] centroid modifie:{}".format(newCentroid))
		self.setCentroid([int(newCentroid[0]),int(newCentroid[1])])
		self.setTempPredictedCentroid([int(newCentroid[0]),int(newCentroid[1])])

	def updateDirection(self):
		size = len(self.centroidBatch)
		if(size>2):
			# self.addDebug("centroides utilises pour la direction : \n {} \n {} \n".format(
			# 	self.centroidBatch[size-1],self.centroidBatch[size-2]))
			newDirection = np.array(self.centroidBatch[size-1]-self.centroidBatch[size-2])
			if ((newDirection != [0,0]).all()):
				newDirection=newDirection/np.linalg.norm(newDirection)
				# print("[DEBUG] new direction : {}".format(newDirection))
				self.setDirection(newDirection)

	def checkDirection(self,x,y):
		return True
		# print("[DEBUG] Object avec ID {} et direction {}".format(self.objectID,
		# 				self.direction))
		# if((self.direction==[0,0]).all()):
		# 	return True
		# xtmp,ytmp=self.centroidBatch[len(self.centroidBatch)-1]
		# newDir=np.array([xtmp-x,ytmp-y])
		# newDir=newDir/np.linalg.norm(newDir)
		#
		# #TODO : ameliorer cette condition :
		# # -dependance a un parametre
		# # -modele dynamique d'evolution
		# if(self.direction.dot(newDir)<0):
		# 	# print("[DEBUG] Blocage pour direction non permise")
		# 	return False
		# return True

	def updateSpeed(self):
		size = len(self.centroidBatch)
		if size >2:
			parcours = (self.centroidBatch[size-2]-self.centroidBatch[size-1])
			self.speed = np.linalg.norm(parcours)

	### Predict-specific methods

	def predictNext(self):
		if(self.getSpeed()==None or (self.getDirection()==None).any()):
			newPred=self.getCentroid()
		else:
			newPred = self.getTempPredictedCentroid() + self.getSpeed()*self.getDirection()
		# print("[DEBUG] : tracage des centroids predits pour l'objet {}".format(self.getID()))
		# print(" centroid actuel     : {}".format(self.getCentroid()))
		# print(" centroide predit    : {}".format(newPred))
		# print(" vitesse 			: {}".format(self.getSpeed()))
		# print(" direction 			: {}".format(self.getDirection()))
		newPred=np.array([int(newPred[0]),int(newPred[1])])
		# self.addDebug("Centroide predit : {}\n".format(newPred))
		self.setPredictedCentroid(newPred)
		self.setTempPredictedCentroid(newPred)



	### Searching for bugs

	def addDebug(self,text):
		self.debugText=self.debugText+text

	def getDebug(self):
		header="ID de l'objet : {} \n".format(self.getID())
		header = header + "centroide : {}\n".format(self.getCentroid())
		header = header + "disparition : {}\n".format(self.getDisapeared())
		header = header + "centroidBatch : {}\n".format(self.getCentroidBatch())
		header = header + "centroide predit : {}\n".format(self.getPredictedCentroid())
		return header + self.debugText
