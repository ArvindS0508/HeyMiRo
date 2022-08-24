import nlp
from nlp_bigram_ver import parse
import sys
import speech_recognition as sr
import os
from os import path


# currently only tests left, right and forward operations
test_input = [
             "go left and then right",
             "go left then right then forward",
             "do not go left and instead go right",
             "after going left go forward and then right",
             "do not go left and do not go right only go forward",
             "go forward and then right not left",
             "once you have made a left turn go forward and then turn right",
             "turn left and then right but don't move forward",
             "once you've gone left go right",
             "turn right move forward then turn left",
             "turn left turn left again move forward and then turn right and right again",
             "turn left then right then left then right then do not go forward",
             "move forward then turn left move forward again and then turn right and move forward again",
             "turn right go forward and then go forward some more",
             "move forward then make a left turn and go forward without turning right",
             "before turning left go right",
             "you're amazing",
             "you are very good and I like you",
             "I think it is very impressive",
             "you are bad",
             "I hate you",
             "the weather is very nice today",
             "before going left go right",
             "before going left and right go forward",
             ]

test_output_expected_base = [
                       ['left', 'forward', 'right', 'forward'],
                       ['left', 'forward', 'right', 'forward', 'forward'],
                       ['right', 'forward'],
                       ['left', 'forward', 'forward', 'right', 'forward'],
                       ['forward'],
                       ['forward', 'right', 'forward'],
                       ['left', 'forward', 'right'],
                       ['left', 'right'],
                       ['left', 'forward', 'right', 'forward'],
                       ['right', 'forward', 'left'],
                       ['left', 'left', 'forward', 'right', 'right'],
                       ['left', 'right', 'left', 'right'],
                       ['forward', 'left', 'forward', 'right', 'forward'],
                       ['right', 'forward', 'forward'],
                       ['forward', 'left', 'forward'],
                       ['right', 'forward', 'left'],
                       ['wagtail'],
                       ['wagtail'],
                       ['wagtail'],
                       ['ear'],
                       ['ear'],
                       ['wagtail'],
                       ['right', 'forward', 'left', 'forward'],
                       ['forward', 'left', 'forward', 'right', 'forward'],
                       ]

test_output_expected_red = [
                       ['left', 'forward', 'right', 'forward'],
                       ['left', 'forward', 'right', 'forward', 'forward'],
                       ['right', 'forward'],
                       ['left', 'forward', 'forward', 'right', 'forward'],
                       ['forward'],
                       ['forward', 'right', 'forward'],
                       ['left', 'forward', 'right'],
                       [],
                       ['left', 'forward', 'right', 'forward'],
                       ['right', 'forward', 'left'],
                       ['left', 'left', 'forward', 'left', 'left'],
                       [],
                       ['forward', 'left', 'forward', 'right', 'forward'],
                       ['right', 'forward', 'forward'],
                       ['forward', 'left', 'forward'],
                       ['right', 'forward', 'left'],
                       ['wagtail'],
                       ['wagtail'],
                       ['wagtail'],
                       ['ear'],
                       ['ear'],
                       ['wagtail'],
                       ['right', 'forward', 'left', 'forward'],
                       ['forward', 'left', 'forward', 'right', 'forward'],
                       ]

flag = 0
# red_flag = True
# print_flag = False
# maxsim = 0.1


if __name__=="__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "text":
            print("enter your sentence:")
            test_input = [input()]
            print("enter the expected output:")
            test_output_expected_base = [input().split(' ')]
            test_output_expected_red = test_output_expected_base
        if sys.argv[1] == "audio":
            audiofile = input("enter audio file name:\n")
            #print(path.dirname(path.realpath(__file__)))
            AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), audiofile)
            r = sr.Recognizer()
            with sr.AudioFile(AUDIO_FILE) as source:
                audio = r.record(source)  # read the entire audio file

            # recognize speech using Sphinx
            try:
                transcription = r.recognize_sphinx(audio)
            except sr.UnknownValueError:
                print("Sphinx could not understand audio")
            except sr.RequestError as e:
                print("Sphinx error; {0}".format(e))
            
            test_input = [transcription]
            test_output_expected_base = ["invalid"]
            test_output_expected_red = test_output_expected_base
        if sys.argv[1] == "mic":
            r = sr.Recognizer()
            print("adjusting for ambient noise, please do not talk...")
            with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=5)
            print("ambient noise adjustment complete")
            with sr.Microphone() as source:
                print("Say something!")
                audio = r.listen(source)
            # recognize speech using Sphinx
            try:
                print("recognizing audio...")
                transcription = r.recognize_sphinx(audio)
                print("you said: " + transcription)
            except sr.UnknownValueError:
                print("could not understand audio")
            except sr.RequestError as e:
                print("error; {0}".format(e))
            test_input = [transcription]
            test_output_expected_base = ["invalid"]
            test_output_expected_red = test_output_expected_base
    else:
        red_flag = True
        print_flag = False
        maxsim = 0.1
    if len(sys.argv) > 2:
        print_flag = sys.argv[2] == 'True'
    else:
        red_flag = True
        print_flag = False
        maxsim = 0.1
    if len(sys.argv) > 3:
        red_flag = sys.argv[3] == "True"
    else:
        red_flag = True
        maxsim = 0.1
    if len(sys.argv) > 4:
        maxsim = float(sys.argv[4])
    else:
        maxsim = 0.1

# print(print_flag)
# print(red_flag)
# print(maxsim)

for i in range(len(test_input)):
    inp = test_input[i]
    if red_flag:
        test_output = parse(inp, print_flag=print_flag, red_check=True, maxsimamount=maxsim)
        test_output_expected = test_output_expected_red
    else:
        test_output = parse(inp, print_flag=print_flag, red_check=False, maxsimamount=maxsim)
        test_output_expected = test_output_expected_base
    if test_output != test_output_expected[i]:
        flag += 1
        print("\n", "~"*20)
        print("incorrect for input: ", inp, "\n")
        print("output:\n", test_output, "\n")
        print("expected:\n", test_output_expected[i], "\n")
        print("~"*20)

if flag == 0:
    print("All Clear!")
else:
    print("incorrect total: ", flag)