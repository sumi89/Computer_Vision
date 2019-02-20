import numpy as np
from PIL import Image
import pylab as plt



def transform(H, fp):
    # Transforming point fp according to H
    # Convert to homogeneous coordinates if necessary
    if fp.shape[0] == 2:
        t = np.dot(H, np.vstack((fp, np.ones(fp.shape[1]))))
    else:
        t = np.dot(H, fp)
    return t[:2]


im2 = np.array(Image.open('/Users/sumi/python/computer_vision/banner_small.jpg'), dtype=np.uint8)
plt.figure(1)
plt.imshow(im2)
plt.show(block=False)

source_im = np.array(Image.open('/Users/sumi/python/computer_vision/tennis.jpg'), dtype=np.uint8)
plt.figure(2)
plt.imshow(source_im)
plt.show(block=False)

im2_max_row = im2.shape[0] - 1
im2_max_col = im2.shape[1] - 1
x1 = [0, 0, im2_max_row]
y1 = [0, im2_max_col, 0]
fp1 = np.vstack((x1, y1))
x2 = [0, im2_max_row, im2_max_row]
y2 = [im2_max_col, 0, im2_max_col]
fp2 = np.vstack((x2, y2))



print("Click destination points, top-left, top-right, bottom-left and bottom-right corners")
tp = np.asarray(plt.ginput(n=4), dtype=np.float).T
tp = tp[[1, 0], :]
tp1 = tp[:, :3]
tp2 = tp[:, 1:]

# Using pseudoinverse
# Generating homogeneous coordinates
fph1 = np.vstack((fp1, np.ones(fp1.shape[1])))
fph2 = np.vstack((fp2, np.ones(fp2.shape[1])))
tph1 = np.vstack((tp1, np.ones(tp1.shape[1])))
tph2 = np.vstack((tp2, np.ones(tp2.shape[1])))
H1 = np.dot(tph1, np.linalg.pinv(fph1))
H2 = np.dot(tph2, np.linalg.pinv(fph2))

print((transform(H1, fp1) + .5).astype(np.int))
print((transform(H2, fp2) + .5).astype(np.int))

# Generating pixel coordinate locations
ind = np.arange(im2.shape[0] * im2.shape[1])
row_vect = ind // im2.shape[1]
col_vect = ind % im2.shape[1]
coords = np.vstack((row_vect, col_vect))

m = -im2_max_col/im2_max_row
select = -coords[1]-m*(im2_max_row-coords[0]) >= 0
coords1 = coords[:, select]
coords2 = coords[:, ~select]

new_coords1 = transform(H1, coords1).astype(np.int)
new_coords2 = transform(H2, coords2).astype(np.int)
target_im = source_im
target_im[new_coords1[0], new_coords1[1], :] = im2[coords1[0], coords1[1], :]
target_im[new_coords2[0], new_coords2[1], :] = im2[coords2[0], coords2[1], :]


plt.figure(3)
plt.imshow(target_im)
plt.show(block=True)