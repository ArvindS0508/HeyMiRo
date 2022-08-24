from tabnanny import verbose
import speech_recognition as sr
import pocketsphinx
from pocketsphinx import LiveSpeech
import nlp
from nlp import parse

print("hello")

#hmm = 'en-us'
#lm = 'en-us.lm.bin'
#dic = 'cmudict-en-us.dict'
#
#speech = LiveSpeech(
#    verbose=False,
#    sampling_rate=8000,
#    buffer_size=5096,
#    no_search=False,
#    full_utt=False,
#    hmm=hmm,
#    lm=lm,
#    dic=dic
#)
#
#for phrase in speech:
#    print(phrase)

r = sr.Recognizer()

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

print("adjusting for ambient noise, please do not talk")
with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=5)
print("ambient noise adjustment complete")
while True:
    with sr.Microphone() as source:
        print("Say something!")
        #r.adjust_for_ambient_noise(source, duration=5)
        audio = r.listen(source)

    # recognize speech using Sphinx
    try:
        transcription = r.recognize_sphinx(audio)
        print("you said " + transcription)
        if transcription == "goodbye": break # debug
        comms = parse(transcription, print_flag=False)
        print(comms)
    except sr.UnknownValueError:
        print("could not understand audio")
    except sr.RequestError as e:
        print("error; {0}".format(e))