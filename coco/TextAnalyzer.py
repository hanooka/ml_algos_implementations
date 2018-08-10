import threading
import warnings
warnings.filterwarnings("ignore", message="detected Windows; aliasing chunkize to chunkize_serial")
import gensim.models.keyedvectors as word2vec


class TextAnalyzer:
    __instance = None

    @staticmethod
    def get_instance():
        if TextAnalyzer.__instance is None:
            TextAnalyzer()
        return TextAnalyzer.__instance

    def __init__(self):
        if TextAnalyzer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            super(TextAnalyzer, self).__init__()
            TextAnalyzer.__instance = self
            self.stop_words = set()
            # self.word2vec = word2vec.KeyedVectors.load_word2vec_format('.//GoogleNews-vectors-negative300.bin', binary=True)

    def parse(self, msg):
        msg = self.filter_letters(msg)
        msg_tokens = self.tokenize(msg)
        return msg_tokens

    def remove_stop_words(self, msgTokens):
        if len(self.stopWords) == 0:
            import nltk
            nltk.data.path
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        return [w for w in msgTokens if w not in self.stopWords]

    def filter_letters(self, msg):
        return ''.join([c for c in msg if c.isalpha() or c == ' '])

    def tokenize(self, msg):
        return msg.split(' ')

    def similarity_score(self, vec1, vec2):
        return 1