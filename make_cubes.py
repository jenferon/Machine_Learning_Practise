import numpy as np
import py21cmfast as p21c
from scipy import stats
import matplotlib.pyplot as plt

Tvir = [4.4,5.6,4.699,5.477,4.8]
eta = [131.341,19.037,30.0,200.0,131.341]
box_dimms = 128.0
HII_DIM = 64





for ii, tt in enumerate(Tvir):
	coeval = p21c.run_coeval(redshift=12.0,user_params={"HII_DIM":HII_DIM, "BOX_LEN":box_dimms}, flag_options= {"USE_TS_FLUC": True}, astro_params={"HII_EFF_FACTOR":eta[ii], "ION_Tvir_MIN":tt})

	tb = np.array(coeval.brightness_temp)
	
	for jj in range(0,len(tb)):
		to_save = (tb[:,:,jj]-np.min(tb[:,:,jj]))/(np.max(tb[:,:,jj])-np.min(tb[:,:,jj]))
		np.save("coeval_21cmfast/Tb_coeval_z12_Tvir_eta_{}_{}_{}x".format(tt,eta[ii],jj), to_save)
		to_save = (tb[:,jj,:]-np.min(tb[:,jj,:]))/(np.max(tb[:,jj,:])-np.min(tb[:,jj,:]))
		np.save("coeval_21cmfast/Tb_coeval_z12_Tvir_eta_{}_{}_{}y".format(tt,eta[ii],jj), to_save)
		to_save = (tb[jj,:,:]-np.min(tb[jj,:,:]))/(np.max(tb[jj,:,:])-np.min(tb[jj,:,:]))
		np.save("coeval_21cmfast/Tb_coeval_z12_Tvir_eta_{}_{}_{}z".format(tt,eta[ii],jj), to_save)
		"""fig, ax = plt.subplots(1,1)
		im = ax.imshow(tb[:,:,jj])
		plt.colorbar(im, ax=ax)
		plt.savefig("/home/ppxjf3/coeval_21cmfast/input_image{}_{}.png".format(ii,jj), dpi=330)
		plt.close()"""
	
	#fits.writeto("coeval_21cmfast/Tb_coeval_z12_Tvir{}_eta{}.fits".format(tt,eta[ii]),tb,overwrite=True)
	print("finished run {} with Tvir={}, eta={}".format(ii,tt,eta[ii]))
fig, ax = plt.subplots(1,1)
im = ax.imshow(to_save)
plt.colorbar(im, ax=ax)
plt.savefig("/home/ppxjf3/coeval_21cmfast/input_image.png", dpi=330)
plt.close()
"""
filename = 'Tb_coeval_z12.h5'
coeval.save(filename, 'coeval_21cmfast/')


filename = "coeval_21cmfast/Tb_coeval_z12.h5"
h5 = h5py.File(filename,'r')

indata = h5.get('dataset_name').value

h5.close()

print(indata.shape)
indata = np.array(indata)
"""
