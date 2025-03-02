import numpy as np
import matplotlib.pyplot as plt
import psd

def psa(prediction, ground_truth):

    prediction_ps = psd.get_average_spectrum(prediction)
    ground_truth_ps = psd.get_average_spectrum(ground_truth)

    print(np.shape(prediction_ps))
    plt.plot(prediction_ps[0], prediction_ps[1])
    plt.show()

if __name__=="__main__":

    # Load ground truth and prediction
    save_path_tr = 'trajectories/lorenz63_tr.npy'
    save_path_gt = 'trajectories/lorenz63_gt.npy'
    prediction, ground_truth = np.load(save_path_tr), np.load(save_path_gt)

    # Analysis
    psa(prediction, ground_truth)


