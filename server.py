import pandas as pd
from flask import Flask, request, jsonify
import pickle
from resemblyzer import VoiceEncoder, preprocess_wav
import warnings
import torch
from ann import ANN
import soundfile as sf

encoder = VoiceEncoder()
warnings.filterwarnings("ignore")


#model=torch.load('/Users/saudshranr/Downloads/ann-100q-97.6.pth', map_location="cpu")
model = ANN(input_dim=256, output_dim=100)
model.load_state_dict(torch.load('/Users/fasilsaeeud/Desktop/deploy_new/ann-100q-97.6.pth', map_location=torch.device('cpu')))


app = Flask(__name__)

#model = torch.load('/home/saudda/mysite/ann.bin')

#with open('/home/saudda/mysite/ann.bin', 'rb') as f:
    #model = pickle.load(f)

#model=pickle.load('home/saudda/mysite/ann.bin','rb')

def extract(fname):
    signal, sr = sf.read('try1.wav')
    # print(signal.float())
    signal = encoder.embed_utterance(torch.tensor(signal))
    return signal

@app.route('/pre',methods=['GET','POST'])
def predict():

    logdata = request.stream.read()

    with open('try1.mp3', 'wb') as f:
        f.write(logdata)



    f = extract('try1.wav')
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(f[None, :]))


    top_p, top_class = prediction.topk(1, dim=1)
    df=pd.read_csv('/Users/fasilsaeeud/Desktop/deploy_new/readers_100.csv')
    id=top_p.item()
    data = df[df['class_id'] == id]
    name = data['qari_ar']
    out= str(name.item())+'_'+ str(top_class.item())
    print(out)
    return out



@app.route('/',methods=['GET','POST'])
def hello():
    return 'hhhhhh'



