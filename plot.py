import numpy as np
import matplotlib.pyplot as plt

#Timepoints
nrTimepoints = 20000


Angle1 = np.loadtxt('Angle1')
Angle2 = np.loadtxt('Angle2')
Latent = np.loadtxt('latent_PCA_ADP')

print(np.average(Angle1))
print(np.average(Angle2))

Angle1 = -Angle1
Angle2 = -Angle2

Angle1 = Angle1+3.14159
Angle2 = Angle2+3.14159

Angle1[Angle1>3.14159] = Angle1[Angle1>3.14159] - 3.14159*2
Angle2[Angle2>3.14159] = Angle2[Angle2>3.14159] - 3.14159*2


plt.figure()
plt.plot(np.linspace(1,nrTimepoints,nrTimepoints),Angle1)
plt.plot(np.linspace(1,nrTimepoints,nrTimepoints),Angle2)
plt.show()

plt.figure()
plt.axis([-3.14,3.14,-3.14,3.14])
#plt.scatter(Angle1,Angle2,s=1)
plt.scatter(Angle1,Angle2,s=1,c=Latent)
plt.show()

