import numpy as np

f = 'test.npy'
dat=np.load(f)

ret = np.argmax(dat, axis=1)
print(list(ret))



# import matplotlib.pyplot as plt
# import numpy as np
#
# for i in range(10):
#     f = f'/Users/yuting/.cache/comvis/imgcls/test/img/test_{i}.npy'
#     dat = np.load(f)
#     plt.imshow(dat)
#     plt.show()