import numpy as np
import math
M = []
for i in range(50):
    for j in range(50):
        for m in range(50):
            for n in range(50):
                M.append(math.pow(i-m,2)+math.pow(j-n,2))
M = np.reshape(M, (2500,2500))
np.savetxt("testOutput.csv", M, delimiter=",")
