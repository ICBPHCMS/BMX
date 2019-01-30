import os
import argparse

import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt

# importing my local files
from reference_model import get_trained_model
from merge_files import merge_files

# parse arguments to make it steerable
parser = argparse.ArgumentParser()
parser.add_argument( "-p_out", '--out_path', default='/bill1/', 
                    help="output path")
parser.add_argument("-vtx", '--n_vertex', default=1, type=int,
                    help="n_vertex")
parser.add_argument("-p_in", '--in_path',
                    default='/Users/mstoye/work/copy/mutrk',
                    help="input path")
parser.add_argument(    "-class",    '--weight_class',   default=1.,
                    type=float, help="weight class")
parser.add_argument( "-regress", '--weight_regress', default=0.,
                    type=float, help="weight regress")

parser.add_argument("-inv_grad", '--revert_grad', default=False, type=bool,
                    help="invert gratdient")
args = parser.parse_args()

# hide argparser below
revert_grad = args.revert_grad
n_vertex = args.n_vertex
out_path = args.out_path
path_name = args.in_path
loss_weights = [args.weight_class, args.weight_regress]

# if True it retrains, wlse it reads a model
retrain = True  # train the NN (~20 min), otherwise it needs a model from disc.
# name of the model for saving or loading
model_files = 'selu_mode'

# make the output directories
os.mkdir(os.getcwd() + out_path)
out_path = os.getcwd() + out_path

###########################################
# get the data !!! ########################

(X_train, is_MC_train, ref_train, bmass_train, Y_train) = merge_files(
    path_name, 'train', n_vertex=n_vertex)
(X_test, is_MC, ref_test, bmass_test, Y_test) = merge_files(
    path_name, 'test', n_vertex=n_vertex)

# though shall preprocess (var = 1, mean = 0) for gradiend based learning!
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

############################################


# the below weights make sure that MC is not used for the second loss,
# which is the mass regression
background = 1 - is_MC_train
all_events = np.ones(background.shape)
weights = [all_events.ravel(), background.ravel()]


if not retrain:
    # load json and create model
    json_file = open(model_files + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    from keras.models import model_from_json
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_files + '.h5')
    score_NN_test = loaded_model.predict(X_scaled_test)
    score_NN_train = loaded_model.predict(X_scaled_train)

if retrain:
    model = get_trained_model(
        X_scaled_train,
        Y_train,
        bmass_train,
        n_vertex,
        weights,
        loss_weights=loss_weights,
        revert_grad=revert_grad)
    [score_NN_test, res_mass_test] = model.predict(X_scaled_test)
    model_json = model.to_json()
    with open(out_path + model_files + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(out_path + model_files + '.h5')
    print("Saved model to disk")


##################################################
### medel is trained/loaded, now we plot
##################################################

def plot_discriminator(i_vertex, score_NN_test, Y_test, is_MC):

    print 'plots for vertex ', i_vertex
    print 'Signal is the correct vertex only'
    plt.figure(i_vertex)
    hist_a, bins, _ = plt.hist(score_NN_test[:, i_vertex][(
        Y_test[:, i_vertex] < 0.5).ravel()], alpha=0.5, label='background', bins=200)
    hist_b, bins, _ = plt.hist(score_NN_test[:, i_vertex][(
        Y_test[:, i_vertex] > 0.5).ravel()], alpha=0.5, label='signal', bins=bins)
    plt.yscale('log')
    plt.xlabel('score NN')
    plt.legend()
    plt.savefig(out_path + 'score_signal' + str(i_vertex) + '.png')
    # plt.show()
    # using MC as signal
    # print 'Simulation can contain wrong vertex'
    plt.figure(i_vertex + 10)
    plt.hist(score_NN_test[:, i_vertex][(is_MC < 0.5).ravel()],
             alpha=0.5, label='only data', bins=bins)
    plt.hist(score_NN_test[:, i_vertex][(is_MC > 0.5).ravel()],
             alpha=0.5, label='simulation', bins=bins)
    plt.yscale('log')
    plt.xlabel('score NN')
    plt.legend()
    plt.savefig(out_path + 'score_simulation' + str(i_vertex) + '.png')
    plt.show()


for i_vertex in range(n_vertex):

    plot_discriminator(i_vertex, score_NN_test, Y_test, is_MC)

##################################################
### Now we plot the mass regression results 
##################################################

# only use first vertex
x = res_mass_test[:, 0]
y = bmass_test[:, 0]


plt.figure(101)
print 'True mean ', y.mean(
), ' would be best is nothing about mass is learned, but the sample mean'
plt.hist2d(x, y, bins=40, cmin=0.001)
plt.savefig(out_path + 'pred_true_mass_all.png')

plt.figure(102)
x_signal = x[(Y_test[:, 0] > 0.5).ravel()]
y_signal = y[(Y_test[:, 0] > 0.5).ravel()]
plt.hist2d(x_signal, y_signal, bins=40, cmin=0.001)
plt.xlabel('predicted mass [GeV]')
plt.ylabel('true mass [GeV]')
plt.savefig(out_path + 'pred_true_mass_signal.png')

plt.figure(103)
print 'True mean for signal ', y_signal.mean()
plt.hist(x_signal, bins=40)
plt.xlabel('predicted mass [GeV]')
plt.savefig(out_path + 'pred_mass_signal.png')

plt.figure(104)
x_bkg = x[(Y_test[:, 0] < 0.5).ravel()]
y_bkg = y[(Y_test[:, 0] < 0.5).ravel()]
plt.hist2d(x_bkg, y_bkg, bins=40, cmin=0.001)
plt.xlabel('predicted mass [GeV]')
plt.ylabel('true mass [GeV]')
plt.savefig(out_path + 'pred_true_mass_bkg.png')

plt.figure(105)
print 'True mean for for background ', y_bkg.mean()
plt.hist(x_bkg, bins=40)
plt.xlabel('predicted mass [GeV]')
plt.savefig(out_path + 'pred_mass_bkg.png')


sel_logic_sim = (is_MC > 0.5).ravel()
sel_logic_data = (is_MC < 0.5).ravel()
sel_logic_sim_true = np.logical_and(
    (is_MC > 0.5).ravel(), (Y_test[:, 0] > 0.5).ravel())

threshold = []
background_rates = []

for i_vertex in range(n_vertex):
    sel_logic_true = (Y_test[:, i_vertex] > 0.5).ravel()
    cut_ref = ref_test[:, i_vertex][sel_logic_true]
    signal_eff = cut_ref.sum() / cut_ref.shape[0]
    cut_ref_false = ref_test[:, i_vertex][(Y_test[:, i_vertex] < 0.5).ravel()]
    #print cut_ref_false.sum()/cut_ref_false.shape[0]

    fpr, tpr, thresholds = metrics.roc_curve(
        Y_test[:, i_vertex], score_NN_test[:, i_vertex])
    for i in range(tpr.shape[0]):
        if tpr[i] > signal_eff:
            #print tpr[i]
            threshold.append(thresholds[i])
            print 'background rate ', fpr[i]
            print 'signal rate ', tpr[i]
            background_rates.append(fpr[i])
            break

np.save(out_path + 'background_rates.npy', x)

print 'cust on NN that have same signal efficiency as simple cuts on 5 vars', threshold


def b_mass_plot(i_vertex, thresholds, sel_logic, bins):

    cut_mass_sim = bmass_test[:, i_vertex][sel_logic]
    cut_ref = ref_test[:, i_vertex][sel_logic]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, gridspec_kw={
        'height_ratios': [2, 1, 1]}, figsize=(7, 5))
    ax1.set_xlim(bins[0], bins[-1])
    ax1.grid()
    ax1.set_ylabel('entries')
    ax2.grid()
    score_NN_test_cut = score_NN_test[:, i_vertex][sel_logic]
    my_hist_a, bins_cut, _ = ax1.hist(
        cut_mass_sim[(cut_ref > 0.5).ravel()], bins=NBINS, alpha=0.5, label='cut')
    my_hist_b, bins_cut, _ = ax1.hist(cut_mass_sim[(
        score_NN_test_cut > thresholds[i_vertex]).ravel()], bins=bins_cut, alpha=0.5, label='NN')
    ax2.legend()
    ax2.set_ylabel('ratio')
    ax3.grid()

    # ratio plot
    my_hist_a[my_hist_a == 0] = 0.0001
    #print my_hist_b, ' a ', my_hist_a
    ratio = my_hist_b / my_hist_a
    #print 'ratio',ratio
    ratio_err = np.sqrt(
        np.sqrt(my_hist_b) /
        my_hist_b *
        np.sqrt(my_hist_b) /
        my_hist_b +
        np.sqrt(my_hist_a) /
        my_hist_a *
        np.sqrt(my_hist_a) /
        my_hist_a)

    ax2.plot(bins_cut[:-1] + 0.5 * (bins_cut[1] - bins_cut[0]), ratio, 'r+')
    ax2.set_ylim(0., 1.5)
    ax2.set_xlim(bins[0], bins[-1])
    ax2.set_xlabel('bmass [GeV] of vetex no. ' + str(i_vertex))
    diff = my_hist_b - my_hist_a
    ax3.set_xlim(bins[0], bins[-1])
    ax3.plot(bins_cut[:-1] + 0.5 * (bins_cut[1] - bins_cut[0]), diff, 'r+')
    ax3.set_ylabel('diff')
    plt.savefig(out_path + 'bmass' + str(i_vertex) + '.png')
    # plt.show()
    return my_hist_a, my_hist_b, bins_cut


NBINS = 20
bins = np.linspace(5.175, 7, NBINS, endpoint=False)
NBINS = bins
hist_a_0, hist_b_0, bins = b_mass_plot(0, threshold, sel_logic_data, bins)
#hist_a_0, hist_b_0, bins = b_mass_plot(1, threshold,sel_logic_data,bins)
