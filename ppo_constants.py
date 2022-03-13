# Environment
ENV_NAME = "CartPole-v1"
NO_OP_MAX_STEPS = 0
ACTION_REPETITIONS = 1
FRAMES_STACKED = 4
NUM_EPISODES = int(1e6)

# Logging
LOG_DIR = "logs"
EPISODES_PATIENCE = 100

# Training hyperparams
NUM_EPOCHS = 3
BATCH_SIZE = 32
CLIP_GRAD = 1
POLICY_LEARNING_RATE = 0.0001
VALUE_LEARNING_RATE = 0.0001
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.5
ENTROPY_LOSS_COEFF = 0.01
HORIZON = 128
GAMMA = 0.99
LAMBDA = 0.95

# Network architecture
INPUT_SIZE = 40
NUM_FC_HIDDEN_UNITS = 128