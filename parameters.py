MODEL_WEIGHT = 'model/model_12_13_16.h5'

SPEECH_FILE = '/home/xiao/projects/speech_enhacement_spectral_domain/data/train/*.wav'
NOISE_FILE = '/home/xiao/projects/speech_enhacement_spectral_domain/data/noise/*.wav'
RESULT_MODEL = 'model/model_12_22_16.h5'

TEST_SPEECH_FILE = '/home/xiao/projects/speech_enhacement_spectral_domain/data/test/4014-186175-0000.wav'
TEST_NOISE_FILE = '/home/xiao/projects/speech_enhacement_spectral_domain/data/noise/restaurant3_sub.wav'

RESULT_DIR = 'result/'


NB_EPOCH = 100000
BATCH_SIZE = 20

no_frames = 8 # even number!
numCep = 93
frame_step = 256
frame_len = 512
fft_len = 512

noise_aware_frame_lenght = 10

MIN_SNR = 15
MAX_SNR = 20

SNR = 5

layer1_dimention = 2800
layer2_dimention = 2800
layer3_dimention = 2800