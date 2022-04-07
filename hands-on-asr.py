import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio


from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav')

recording = 'potatoes.wav'
asr_model.transcribe_file(recording)