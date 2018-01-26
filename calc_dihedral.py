import sys
import numpy as np
import re

#Import Name
inputName = 'target_trj_autoCNN_ADP'
#inputName = 'cleaned_input_ADP'

#Period
period = 22

#Nr of timepoints
nrTimepoints = 20000

PI = np.arccos(-1.)

#def get_distance_components(a,b,cell):
#    '''
#    This subroutine calculate the components of distance between two vectores w pbc
#    '''
#    v=np.zeros(3)
#    v[0]=a[0]-b[0] - np.around((a[0]-b[0])/cell[0])*cell[0]
#    v[1]=a[1]-b[1] - np.around((a[1]-b[1])/cell[1])*cell[1]
#    v[2]=a[2]-b[2] - np.around((a[2]-b[2])/cell[2])*cell[2]
#    return v


#def calculate_dihedral(a,b,c,d,cell):
def calculate_dihedral(a,b,c,d,cell):
    '''
    This subrountine calculate the dihedral angles between 4 atoms.
    It uses the position of the 4 atoms 
    '''
#    v1 = get_distance_components(a,b,cell)
#    v2 = get_distance_components(b,c,cell)
#    v3 = get_distance_components(c,d,cell)

    v1 = a-b
    v2 = b-c    
    v3 = c-d

    p1 = np.cross(v1,v2)
    p2 = np.cross(v3,v2)

    l1 = np.cross(p1,p2)
    l2 = np.dot(l1,v2/np.linalg.norm(v2))
    l3 = np.dot(p1,p2)

    theta = np.arctan2(l2,l3)
    #theta = np.arctan2(l3,l2)
    return theta


atom_ids1 = [4,6,8,14]
atom_ids2 = [6,8,14,16]

Angle1 = np.ones(nrTimepoints)
Angle2 = np.ones(nrTimepoints)

traj = np.loadtxt(inputName)
traj = traj[:,0:3]

for i in range(0,nrTimepoints):
	Angle1[i] = calculate_dihedral(traj[period*i+atom_ids1[0],:],traj[period*i+atom_ids1[1],:],traj[period*i+atom_ids1[2],:],traj[period*i+atom_ids1[3],:], [100,100,100])
	Angle2[i] = calculate_dihedral(traj[period*i+atom_ids2[0],:],traj[period*i+atom_ids2[1],:],traj[period*i+atom_ids2[2],:],traj[period*i+atom_ids2[3],:], [100,100,100])



np.savetxt('Angle1',Angle1)
np.savetxt('Angle2',Angle2)

