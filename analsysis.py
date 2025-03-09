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

def visualize_ps(prediction, prediction_ps, ground_truth_ps, dimension):

    # Frequencies
    freqs = np.fft.rfftfreq(np.shape(prediction)[0])

    for i in range(dimension):

        # Plot Prediction and ground truth
        plt.figure(figsize=(6, 4))
        plt.plot(freqs, prediction_ps[i], label=f"Predicted time series dimension {i+1}" )
        plt.plot(freqs, ground_truth_ps[i], label=f"Ground truth time series dimension {i+1}", color="green", linestyle="-")

        # Plotting
        plt.xlabel("Frequency")
        plt.xlim(0, 0.05)
        plt.ylabel("Power")
        plt.title("Power Spectrum of Generated Trajectories")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__=="__main__":

    # Load ground truth and prediction
    save_path_tr = 'trajectories/lorenz63_tr.npy'
    save_path_gt = 'trajectories/lorenz63_gt.npy'
    dimension = 3
    prediction, ground_truth = np.load(save_path_tr), np.load(save_path_gt)

    # Analysis
    prediction_ps, ground_truth_ps, ps_error = psa(prediction, ground_truth)

    # Plot
    visualize_ps(prediction, prediction_ps, ground_truth_ps, dimension)


