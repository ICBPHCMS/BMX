import os
import fnmatch
import numpy as np


def features_from_vertex(n_vertex, open_file):
    features = []
    for i in range(n_vertex):
        features.append(np.copy(open_file['features'][:, i, :]))
    return tuple(features)


def one_hot_labels(n_vertex, gen_index):
    gen_label = []
    for i in range(n_vertex):
        gen_label.append(gen_index[:, i])
    add_one = 1 - gen_index[:, 0:n_vertex + 1].sum(axis=1)
    gen_label.append(add_one)
    return np.vstack(gen_label).transpose()


def merge_files(folder, filename, n_vertex=2, B_mass_0_min=5.175):
    '''
        File need to be merged
        folder: path to files
        filename: is a unique string in the name of the files to be merged
        assumes .npz format
        '''

    print 'Reading files from \"', folder, '\" which include the name \"', filename, '\" and use ', n_vertex, ' vertices and pu them into flat vector, the first vertex mass cut is ', B_mass_0_min

    files = os.listdir(folder)
    read = False  # flags if the first file was read

    for file in files:
        if filename in file:
            open_file = np.load(folder + '/' + file)  # gets the file
            if not read:
                read = True

                # copy do get the file into memory, opening is not enough
                features = np.hstack(features_from_vertex(n_vertex, open_file))
                #print "feature shape ", open_file['features'].shape
                truth = np.copy(open_file['truth'])
                refSel = np.copy(open_file['refSel'])
                bmass = np.copy(open_file['bmass'])
                gen_index = np.copy(open_file['genIndex'])
            #print 'that is the keys of the recarray:', open_file.keys()

            else:
                new_feature = np.hstack(
                    features_from_vertex(
                        n_vertex, open_file))
                features = np.concatenate((features, new_feature))
                truth = np.concatenate((truth, np.copy(open_file['truth'])))
                refSel = np.concatenate((refSel, np.copy(open_file['refSel'])))
                bmass = np.concatenate((bmass, np.copy(open_file['bmass'])))
                gen_index = np.concatenate(
                    (gen_index, np.copy(open_file['genIndex'])))

# delete nan entries and apply cuts
    kill_crap = np.isnan(features).any(axis=1)
    kill_crap = np.logical_or((bmass[:, 0] < B_mass_0_min).ravel(), kill_crap)
    features = features[(kill_crap < 0.5).ravel()]
    truth = truth[(kill_crap < 0.5).ravel()]
    refSel = refSel[(kill_crap < 0.5).ravel()]
    bmass = bmass[(kill_crap < 0.5).ravel()]
    gen_index = gen_index[(kill_crap < 0.5).ravel()]

    return features, truth, refSel, bmass[:, 0:n_vertex], one_hot_labels(
        n_vertex, gen_index)
