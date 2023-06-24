import numpy as np
kpts = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
boxes = np.array([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]])
print(zip(kpts,boxes))
for person_kpts, person_box in zip(kpts, boxes):
    x1, y1, x2, y2 = person_box[:4]
    dets = np.array(person_box[:5])[np.newaxis,:]
    print(dets)
