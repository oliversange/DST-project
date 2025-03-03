import numpy as np
import matplotlib.pyplot as plt
import psd

def psa(prediction, ground_truth):

    # Compute power spectra
    prediction_ps = psd.get_average_spectrum(prediction.T)
    ground_truth_ps = psd.get_average_spectrum(ground_truth.T)

    # Compute error
    print(np.shape(prediction))
    print(np.shape(ground_truth))
    ps_error = psd.power_spectrum_error(np.reshape(prediction, (1, np.shape(prediction)[0], np.shape(prediction)[1])), np.reshape(ground_truth, (1, np.shape(prediction)[0], np.shape(prediction)[1])))
    print(f"The power spectrum error is: {ps_error}")

    return prediction_ps, ground_truth_ps, ps_error

def visualize_ps(prediction, prediction_ps, ground_truth_ps):

    # Frequencies
    freqs = np.fft.rfftfreq(np.shape(prediction)[0])
    print(np.shape(freqs))

    # Unpack
    pred_x, pred_y, pred_z = prediction_ps[:3]
    gt_x, gt_y, gt_z = ground_truth_ps[:3]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, pred_y, label="Reconstruction", color="r", linestyle="-")
    plt.plot(freqs, gt_y, label="Ground truth")
    #plt.plot(freqs, pred_y, label="Y-Dimension", color="g", linestyle="--")
    #plt.plot(freqs, pred_z, label="Z-Dimension", color="b", linestyle=":")

    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.title("Power Spectrum of Generated Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":

    # Load ground truth and prediction
    save_path_tr = 'trajectories/lorenz96_tr_plus.npy'
    save_path_gt = 'trajectories/lorenz96_gt_plus.npy'
    prediction, ground_truth = np.load(save_path_tr), np.load(save_path_gt)

    # Analysis
    prediction_ps, ground_truth_ps, ps_error = psa(prediction, ground_truth)

    # Plot
    visualize_ps(prediction, prediction_ps, ground_truth_ps)


