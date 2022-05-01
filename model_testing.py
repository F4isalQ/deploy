from locale import normalize
import librosa
from numpy import float32
import sounddevice as sd
import resemblyzer
import soundfile as sf
import torchaudio as tu
import torch
from ann import ANN


def extract(fname):
    signal, sr = tu.load(fname)
    signal = _resample_if_necessary(signal, sr)
    signal = _mix_down_if_necessary(signal)
    signal = _cut_if_necessary(signal)
    signal = _right_pad_if_necessary(signal)
    signal = transformation(signal)
    # print('before: ', signal.shape)
    if DELTA:
        delta1 = tu.functional.compute_deltas(signal)
        delta2 = tu.functional.compute_deltas(delta1)
        signal = tu.concat([signal, delta1, delta2], dim=1)
    signal = torch.mean(torch.tensor(signal), dim=2)
    signal = torch.flatten(signal)
    return signal


def _cut_if_necessary(signal):
    if signal.shape[1] > NUM_SAMPLES:
        signal = signal[:, :NUM_SAMPLES]
    return signal


def _right_pad_if_necessary(signal):
    length_signal = signal.shape[1]
    if length_signal < NUM_SAMPLES:
        num_missing_samples = NUM_SAMPLES - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal


def _resample_if_necessary(signal, sr):
    if sr != SAMPLE_RATE:
        resampler = tu.transforms.Resample(sr, SAMPLE_RATE)
        resampler.to(device)
        signal = resampler(signal)
    return signal


def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def record():
    samplerate = 22050  # samples per second
    duration = 15  # number of seconds
    filename = 'output.wav'

    print('recording now...')
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    # the line below you specify the path to the recorded file
    sf.write("/Users/fasilsaeeud/Desktop/deploy_new/new_aud" + filename, audio.astype(float32), samplerate)
    #librosa.output.write_wav("C:\\Users\\ASUS\\Desktop\\" + filename, audio, samplerate)
    

def predict(model, signal):
    model.eval()
    with torch.no_grad():
        prediction = model(signal)

    #print(prediction)
    top_p, top_class = prediction.topk(1, dim=1)
    return top_p, top_class

if __name__ == '__main__':
    SAMPLE_RATE = 22050
    NUM_SAMPLES = SAMPLE_RATE * 15
    DELTA = False
    transformation = tu.transforms.MFCC(n_mfcc=40, sample_rate=SAMPLE_RATE, melkwargs={'n_mels': 64})
    if torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cpu"
    print(f"Using {device}")

    model = ANN(input_dim=256, output_dim=100)
    model.load_state_dict(torch.load('/Users/fasilsaeeud/Desktop/deploy_new/ann-100q-97.6.pth', map_location=torch.device('cpu')))

    encoder = resemblyzer.VoiceEncoder()

