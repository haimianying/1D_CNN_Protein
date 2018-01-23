import numpy as np
import re


#Script to read in a modified or autoencoded trajectory and turn it into VMD-readable trj.
#Very primitive, but will do for now.


#Period = repeating period of gro - to clean the box size, atom number and title lines from trajectory.
periodInput = 128

#Period = repeating period of gro - to clean the box size, atom number and title lines from trajectory.
periodReference = 106
referenceExtraLines = 3

#Number of spurious lines
#Automate this?
nRemove = 128-106

#Box size
box = [5.0,5.0,5.0]

#Trj to open in gro format to use as a template
templateGro = '../deca_alanine/nvt_full.gro'

#Autoencoded or modified trj to expand.
trjIn = '../target_trj'

#Output file name
reconstructedTrj = '../modified_original.gro'



#---------------------------------#

output = open(reconstructedTrj,'w')
template = open(templateGro,'r')
#inputTrj = open(trjIn,'r')

i=1
j=1
rmv=False
with open(trjIn) as trj:

	#Add the header
	output.write(template.readline())
	output.write(template.readline())

	for line in trj:
		if not ((i%periodInput > periodInput-nRemove) or (i%periodInput == 0)):
			if not ((j%periodReference > periodReference-referenceExtraLines) or (j%periodReference == 0)):

				prefixes = template.readline().split()
				coords = line.split()
				
				match = re.match(r"([0-9]+)([a-z]+)", prefixes[0], re.I)
				if match:
					items = match.groups()

				output.write('{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}'.format(int(items[0]),items[1],prefixes[1],int(prefixes[2]),float(coords[0]),float(coords[1]),float(coords[2])))
				output.write('\n')
			
			else:
				output.write(template.readline())
			j=j+1
		i = i+1
output.close()
trj.close()


