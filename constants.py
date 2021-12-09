ENV_NAME = "CartPole-v1"

INPUT_SIZE = 40
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE = 5000
LEARNING_RATE = 0.0001
NUM_EPISODES = 10000
CLIP_GRAD = 1
MIN_START_STEPS = 10000
REPLAY_BUFFER_LEN = 100000
ALPHA = 0.5
BETA0 = 0.4
REPLAY_DELAY = 1
NO_OP_MAX_STEPS = 10
ACTION_REPETITIONS = 4
FRAMES_STACKED = 4
