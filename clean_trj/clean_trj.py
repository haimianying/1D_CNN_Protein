import numpy as np
import re


#Script to strip a gro trajectory of everything except coordinates and velocities.
#Leaves a numpy.loadtxt readable file.
#Very primitive, but will do for now.


#Period = repeating period of gro - to clean the box size, atom number and title lines from trajectory.
period = 25

#Number of spurious lines
nRemove = 3

#Trj to open in gro format
trjname = '../ADP/adp_long_full.gro'

#Output file name
cleanedTrj = '../cleaned_input_ADP'



#---------------------------------#

output = open(cleanedTrj,'w')

i=1
rmv=False
with open(trjname) as trj:
	#remove the header
	trj.readline()
	trj.readline()
	for line in trj:
		if not ((i%period > period-nRemove) or (i%period == 0)):
			floats = re.findall("[+-]?\d+\.\d+", line)
			for fl in floats:
				output.write(str(fl)+"	")
			output.write("\n")
		i = i+1

output.close()
trj.close()


