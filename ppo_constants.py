# Environment
ENV_NAME = "CartPole-v1"
INPUT_SIZE = 40
NO_OP_MAX_STEPS = 0
ACTION_REPETITIONS = 1
FRAMES_STACKED = 4
NUM_EPISODES = int(1e6)

# Logging
LOG_DIR = "logs"
EPISODES_PATIENCE = 128
EVAL_FREQUENCY = 5000
EVAL_EPISODE_COUNT = 10

# Training hyperparams
NUM_EPOCHS = 5
BATCH_SIZE = 32
CLIP_GRAD = 1
POLICY_LEARNING_RATE = 0.0001
VALUE_LEARNING_RATE = 0.0001
CLIP_EPSILON = 0.1
VALUE_LOSS_COEFF = 0.5
ENTROPY_LOSS_COEFF = 0.01
HORIZON = 128
GAMMA = 0.99
LAMBDA = 0.95
NORMALIZE_VALUES = False

# Network architecture
NUM_FC_HIDDEN_UNITS = 128
NUM_CHANNELS = 64
