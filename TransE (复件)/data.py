import csv
relationDict={}
with open('../data/WN182/entity2id.txt') as file:
	lines=file.readlines()
	for line in lines:
		l=line.strip().split('\t')
		relationDict[l[0]]=l[1]
print(relationDict)
newList=[]
with open('../data/WN182/train.txt') as file:
	lines=file.readlines()
	for line in lines:
		l=line.strip().split('\t')
		if (l[0] in relationDict.keys()) and (l[1] in relationDict.keys()):
			newList.append(l[0]+'\t'+l[1]+'\t'+l[2])
file=open('../data/WN182/ttt.txt','w')
for triplet in newList:
	file.write(triplet)
	file.write('\n')
file.close()