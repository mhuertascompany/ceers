import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import h5py

zbins = [3, 4, 5, 6]


def load_catalog_values(cat_file):
    dt = [('ID', 'U6'), ('z', 'float'), ('inc', 'int'), ('azm', 'int'), ('x_or_RA', 'float'), ('y_or_Dec', 'float'),
          ('pos_angle', 'float'), ('mf200w', 'float'), ('rad', 'float'), ('rad_gas', 'float'), ('rad_dm', 'float'),
          ('rad_star', 'float'), ('mass', 'float'), ('mass_gas', 'float'), ('mass_dm', 'float'), ('mass_star', 'float'),
          ('sfr', 'float'), ('metal', 'float'), ('rminax', 'float'), ('rmajax', 'float'), ('minax', 'float'),
          ('majax', 'float'), ('ang', 'float'), ('disk', 'float'), ('disk2', 'float'), ('bulge', 'float'),
          ('sfr10', 'float'), ('sfr50', 'float'), ('sfr100', 'float'), ('sfr200', 'float'), ('sfr1000', 'float'),
          ('kappa', 'float')]

    cat_all = np.loadtxt(cat_file, skiprows=1, dtype=dt)
    columns = [c[0] for c in dt]
    df = pd.DataFrame(cat_all, columns=columns)
    return df


def load_morphological_values(morph_path, morph_idx_path):
    fmorph = h5py.File(morph_path, 'r')

    #Get the HDF5 group
    group_morph = fmorph[list(fmorph.keys())[0]]
    morph = group_morph['table'][()]

    dt = np.dtype([('index', 'int')])
    indexes_morph = np.loadtxt(morph_idx_path, skiprows=1, dtype=dt)
    indexes_morph = np.array(indexes_morph)
    morph_all = morph[indexes_morph['index']]

    df2 = pd.DataFrame(morph_all, columns=list(group_morph['_i_table']))
    fmorph.close()

    return df2


def load_data(seed_shuffle, path, band):
    print("Reading input data...")
    print('------------------------------------')

    for zz in zbins:
        idz = np.load(path + "/id_TNG50_z" + str(int(zz)) + "_64pix_F" + str(band) + "W.npy", allow_pickle=True)
        imz = np.load(path + "/image_TNG50_z" + str(int(zz)) + "_64pix_F" + str(band) + "W.npy")

        z = np.zeros(idz.shape[0], dtype=np.float32) + zz

        print('Redshift = ' + str(zz))
        print('ID shape: ' + str(idz.shape))
        print('D shape: ' + str(imz.shape))

        if zz == zbins[0]:
            im_all = imz
            id_all = idz
            z_all = z
        else:
            id_all = np.append(id_all, idz)
            im_all = np.append(im_all, imz, axis=2)
            z_all = np.append(z_all, z)

    del idz, imz, z

    # Negative values to zero and remove nans
    # im_all[im_all < 0] = 0.0
    # im_all[np.isnan(im_all)] = 0.0

    # Transpose array for input simclr
    # im_all = im_all.reshape(im_all.shape[2],im_all.shape[0],im_all.shape[1])
    im_all = im_all.transpose(2, 0, 1)

    print('------------------------------------')
    print('All redshifts')
    print('ID shape: ' + str(id_all.shape))
    print('D shape: ' + str(im_all.shape))

    if seed_shuffle != 0:
        print('Shuffling data...')
        id_all, im_all, z_all = shuffle(id_all, im_all, z_all, random_state=seed_shuffle)

    # Number of galaxies
    nsel = id_all.shape[0]
    im_all = np.float32(im_all)

    print('------------------------------------')
    print('Input data ready!')
    print('------------------------------------')

    return id_all, im_all, z_all, nsel


def load_noise_data(seed_shuffle, path, band):
    print("Reading input data...")
    print('------------------------------------')

    for zz in zbins:
        idz = np.load(path + "/id_CEERS_z" + str(int(zz)) + "_64pix_F" + str(band) + "W.npy", allow_pickle=True)
        imz = np.load(path + "/image_CEERS_z" + str(int(zz)) + "_64pix_F" + str(band) + "W.npy")

        z = np.zeros(idz.shape[0], dtype=np.float32) + zz

        print('Redshift = ' + str(zz))
        print('ID shape: ' + str(idz.shape))
        print('D shape: ' + str(imz.shape))

        if zz == zbins[0]:
            im_all = imz
            id_all = idz
            z_all = z
        else:
            id_all = np.append(id_all, idz)
            im_all = np.append(im_all, imz, axis=2)
            z_all = np.append(z_all, z)

    del idz, imz, z

    # Negative values to zero and remove nans
    # im_all[im_all < 0] = 0.0
    # im_all[np.isnan(im_all)] = 0.0

    # Transpose array for input simclr
    # im_all = im_all.reshape(im_all.shape[2],im_all.shape[0],im_all.shape[1])
    im_all = im_all.transpose(2, 0, 1)

    print('------------------------------------')
    print('All redshifts')
    print('ID shape: ' + str(id_all.shape))
    print('D shape: ' + str(im_all.shape))

    if seed_shuffle != 0:
        print('Shuffling data...')
        id_all, im_all, z_all = shuffle(id_all, im_all, z_all, random_state=seed_shuffle)

    # Number of galaxies
    nsel = id_all.shape[0]

    im_all = np.float32(im_all)

    print('------------------------------------')
    print('Input data ready!')
    print('------------------------------------')

    return id_all, im_all, z_all, nsel