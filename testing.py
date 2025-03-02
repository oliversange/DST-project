import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import utils
from mpl_toolkits.mplot3d import Axes3D

class Test_transformer:

    def __init__(self, enc_seq_len, output_seq_len):

        self.model_input_len = (enc_seq_len+output_seq_len-1)
        self.enc_seq_len = enc_seq_len
        self.output_seq_len = output_seq_len

    def generate_initial_conidtion(self, data, warm_up_steps, max_offset=2000):

        # Pick random starting index
        max_start = min(len(data) - self.model_input_len, len(data) - max_offset)
        start_idx = np.random.randint(0, max_start + 1)

        # Extract ground truth
        ground_truth = data[(start_idx+self.model_input_len+warm_up_steps):]

        return data[start_idx : start_idx + self.model_input_len], ground_truth

    def generate_test_trajectory(self, test_data_path, model_path):

        # Load data and generate initial conidtion
        warm_up_steps = 512
        test_data = np.load(test_data_path)
        initial_condition, ground_truth = self.generate_initial_conidtion(data=test_data, warm_up_steps=warm_up_steps)
        initial_input = torch.tensor(initial_condition.reshape(self.model_input_len, 1, 3))
        T = np.shape(ground_truth)[0]
        print(f'T = {T}')

        # Load model
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()

        # Do inference
        with torch.no_grad():

            # Start with warm_up
            warm_up = self.inference(
                model=model,
                initial_input=initial_input,
                forecast_window=warm_up_steps
            )

            # Prediction
            prediction = self.inference(
                model=model,
                initial_input=warm_up[-self.model_input_len:, :, :],
                forecast_window=T
            )

        return ground_truth, np.reshape(prediction, (prediction.shape[0],3))

    def inference(self, model: nn.Module, initial_input: torch.Tensor, forecast_window: int) -> torch.Tensor:

        # Store Predictions
        predicted_series = initial_input
        current_step = initial_input

        # Run inference
        for _ in tqdm(range(forecast_window)):

            # Generate model inputs and run model
            src = current_step[-self.model_input_len:-(self.output_seq_len-1), :, :]
            tgt = current_step[-self.output_seq_len:, :, :]
            src_mask = utils.generate_square_subsequent_mask(self.output_seq_len, self.enc_seq_len)
            tgt_mask = utils.generate_square_subsequent_mask(self.output_seq_len, self.output_seq_len)
            prediction = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)

            # Append predictions
            predicted_series = torch.cat((predicted_series, prediction[-1:].clone()), dim=0)
            current_step = torch.cat((current_step, prediction[-1:].clone()), dim=0)[1:]

        return predicted_series[self.model_input_len:]

    def plot_trajectories(self, ground_truth, prediction):

        # Extract data
        x, y, z = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2]
        x_p, y_p, z_p = prediction[:, 0], prediction[:, 1], prediction[:, 2]

        # Create plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=0.5, color='darkblue')
        ax.plot(x_p, y_p, z_p, lw=0.5, color='red')

        # Plotting
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Lorenz Attractor")
        plt.show()

if __name__=="__main__":

    # Initialize
    enc_seq_len = 128
    output_seq_len = 32
    test_model = Test_transformer(enc_seq_len, output_seq_len)

    # Inference
    test_data_path = 'data/lorenz63_test.npy'
    model_path = "saved_models/trained_63_model_new.pth"
    ground_truth, prediction = test_model.generate_test_trajectory(test_data_path, model_path)

    print(f'ground truth shape: {ground_truth.shape}')
    print(f'prediction shape: {prediction.shape}')

    # Plot
    test_model.plot_trajectories(ground_truth, prediction)

    # Save trajectory
    save_path_tr = 'trajectories/lorenz63_tr.npy'
    save_path_gt = 'trajectories/lorenz63_gt.npy'
    np.save(save_path_tr, prediction.numpy())
    np.save(save_path_gt, ground_truth)
