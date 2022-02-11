import numpy as np 
import function
image_data=function.imread_show_image()
# np.save('G:/my_project/Identify-abnormal-behaviors-in-video-images-using-neural-networks/data/ImageData.npy'
# ,image_data)
binary_optical_map=function.Optical_map(image_data)
# np.save('G:/my_project/Identify-abnormal-behaviors-in-video-images-using-neural-networks-/data/binary_optical_map_2.npy'
# ,binary_optical_map)


result=function.Add(image_data,binary_optical_map)
# np.save('G:/my_project/Identify-abnormal-behaviors-in-video-images-using-neural-networks-/data/Result_1.npy'
# ,result)

