import dataset as ds
import utils
from torch.utils.data import DataLoader
import torch
import transformer_timeseries as tst
import numpy as np
from tqdm import tqdm

# Set device and load data
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
training_data = np.load('data/lorenz63_on0.05_train.npy')

# Parameters
batch_size = 64
input_size = 3
enc_seq_len = 128  # length of input given to encoder
dec_seq_len = 64  # length of input given to decoder
output_sequence_length = 32  # target sequence length
dim_val=256
n_encoder_layers=2
n_decoder_layers=2
n_heads=8
dim_feedforward_encoder=512
dim_feedforward_decoder=512
num_predicted_features=3
step_size = 8  # Step size, i.e. how many time steps does the moving window move at each step
batch_first = False
epochs = 6

# Make list indices used to slice the data into training chunks
training_indices = utils.get_indices_input_target(
    num_obs=np.shape(training_data)[0],
    input_len=enc_seq_len,
    step_size=step_size,
    forecast_horizon=0,
    target_len=output_sequence_length
)
training_indices = [(t[0], t[-1]) for t in training_indices]

# Making instance of custom dataset class
training_data = ds.TransformerDataset(
    data=torch.tensor(training_data).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
)

# Making dataloader
training_loader = DataLoader(training_data, batch_size, shuffle=True)

# Initiate model
model = tst.TimeSeriesTransformer(
    input_size=input_size,
    dec_seq_len=dec_seq_len,
    out_seq_len=output_sequence_length,
    batch_first=batch_first,
    dim_val=dim_val,
    n_encoder_layers=n_encoder_layers,
    n_decoder_layers=n_decoder_layers,
    n_heads=n_heads,
    dim_feedforward_encoder=dim_feedforward_encoder,
    dim_feedforward_decoder=dim_feedforward_decoder,
    num_predicted_features=num_predicted_features
).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

for epoch in range(epochs):

    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0.0
    model.train()

    for i, (src, trg, tgt_y) in enumerate(tqdm(training_loader, desc=f"Training Epoch {epoch + 1}")):
        src, trg, tgt_y = src.to(device), trg.to(device), tgt_y.to(device)
        print(np.shape(src))
        print(np.shape(trg))
        optimizer.zero_grad()
        if not batch_first:
            src, trg, tgt_y = src.permute(1, 0, 2), trg.permute(1, 0, 2), tgt_y.permute(1, 0, 2)
        print(f'src after {np.shape(src)}')
        print(f'trg after {np.shape(trg)}')
        src_mask = utils.generate_square_subsequent_mask(output_sequence_length, enc_seq_len).to(device)
        tgt_mask = utils.generate_square_subsequent_mask(output_sequence_length, output_sequence_length).to(device)
        prediction = model(src=src, tgt=trg, src_mask=src_mask, tgt_mask=tgt_mask)
        print(f'prediction shape {np.shape(prediction)}')
        loss = criterion(tgt_y, prediction)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(training_loader):.6f}")

#torch.save(model, "saved_models/trained_63_model_new.pth")