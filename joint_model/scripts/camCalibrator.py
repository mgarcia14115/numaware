import sys
from data_collection.calibri import calibrate

cali_images_dir = sys.argv[1]
rows            = int(sys.argv[2])
cols            = int(sys.argv[3])
save_file       = sys.argv[4]

folder = cali_images_dir

save_file= save_file

mtx, dist = calibrate(
    folder=folder,
    rows=rows,
    cols=cols,
    save_file=save_file
)