"""
Code for running inference with transformer
"""

import torch.nn as nn
import torch
import utils
from tqdm import tqdm
import numpy as np


def run_encoader_decoder_inference(
        model: nn.Module,
        src: torch.Tensor,
        forecast_window: int,
        batch_size = 1,
        batch_first: bool = False
) -> torch.Tensor:

    with torch.no_grad():
        # Dimension of a batched model input that contains the target sequence values
        target_seq_dim = 0
        tgt = src[-1, :, :].unsqueeze(0)

        # Iteratively concatenate tgt with the first element in the prediction
        for _ in tqdm(range(forecast_window - 1), desc="Forecasting", unit="step"):

            # Create masks
            dim_a = tgt.shape[0]

            dim_b = src.shape[0]

            tgt_mask = utils.generate_square_subsequent_mask(
                dim1=dim_a,
                dim2=dim_a,
            )

            src_mask = utils.generate_square_subsequent_mask(
                dim1=dim_a,
                dim2=dim_b,
            )

            # Make prediction
            prediction = model(src, tgt, src_mask, tgt_mask)

            # If statement simply makes sure that the predicted value is
            # extracted and reshaped correctly
            if batch_first == False:

                # Obtain the predicted value at t+1 where t is the last time step
                # represented in tgt
                last_predicted_value = prediction[-1, :, :]

                # Reshape from [batch_size, 1] --> [1, batch_size, 1]
                last_predicted_value = last_predicted_value.unsqueeze(0)

            # Detach the predicted element from the graph and concatenate with
            # tgt in dimension 1 or 0
            tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)

        # Create masks
        dim_a = tgt.shape[0]

        dim_b = src.shape[0]

        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_a,
        )

        src_mask = utils.generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b,
        )

        # Make final prediction
        final_prediction = model(src, tgt, src_mask, tgt_mask)

        return final_prediction
