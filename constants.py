ENV_NAME = "CartPole-v1"
INPUT_SIZE = 40
BATCH_SIZE = 32
N_STEP_RETURNS = 3
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 5000
TARGET_UPDATE = 10
LEARNING_RATE = 0.0001
NUM_EPISODES = 100000
CLIP_GRAD = 1.0
MIN_START_STEPS = 1000
REPLAY_BUFFER_LEN = 10000
ALPHA = 0.75
BETA0 = 0.4
REPLAY_DELAY = 300
NO_OP_MAX_STEPS = 0
ACTION_REPETITIONS = 1
FRAMES_STACKED = 4
LOG_DIR = "logs"
DOUBLE_Q_LEARNING = True
NETWORK_NAME = "Dueling DQN"
NOISY_NETS = True
NOISY_SIGMA_INIT = 0.5
NUM_ATOMS = 51
V_MIN = 0
V_MAX = 100
EPISODES_PATIENCE = 100
NUM_FC_HIDDEN_UNITS = 128
