from SpeechRecognition import SpeechRecognizer
from Command import CommandCenter

if __name__ == "__main__":
    speech_recognizer = SpeechRecognizer()
    command_center = CommandCenter()
    speech_recognizer.start()
    command_center.start()
