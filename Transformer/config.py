

SAMPLE_RATE = 24000


PHONEME_DIR = "/home/diffine/Aleksandar/Transformer/Dataset/LibriSpeech/train-other-500"
QNTS_DIR = "/home/diffine/Aleksandar/Transformer/Dataset/LibriSpeech/train-other-500"
PHONEME_SUFFIX = ".phn.txt"
QNTS_SUFFIX = ".qnt.pt"

BATCH_SIZE = 1
EMBEDDING_DIMENSION = 1024
NUM_HEADS = 16
NUM_LAYERS = 12
FF = 4096
MAX_SEQ_LENGTH = 10240

DROPOUT = 0.1

LR = 5e-4
NUM_EPOCHES = 1000
UPDATES_PER_SAVE = 9000  # bilo 1000
TEMP_SAVE = 100
MODEL_SAVE_DIR = "./checkpoints"
