update_data_file = True
data_file = './data/gray.npy'

BUFFER_SIZE = 100000
BATCH_SIZE = 256

EPOCHS = 100
NOISE_DIM = 100

LEARNING_RATE = 1e-4

NUM_EXAMPLES_TO_GENERATE = 16

checkpoint_frequency = 20
checkpoint_restore = False
checkpoint_dir = './training_checkpoints'
log_dir = './logs'