import gensim.models.keyedvectors as word2vec
import itertools
import time
import numpy as np

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
            self.word2vec = word2vec.KeyedVectors.load_word2vec_format('.//GoogleNews-vectors-negative300.bin', binary=True)

    def parse(self, msg):
        msg = self.filter_letters(msg)
        msg_tokens = self.tokenize(msg)
        return msg_tokens

    def remove_stop_words(self, msgTokens):
        stopWords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
                     "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
                     "did", "do", "does", "doing", "during", "each", "few", "for", "from", "further", "had", "has",
                     "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
                     "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
                     "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
                     "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
                     "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the",
                     "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
                     "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "very",
                     "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's",
                     "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would",
                     "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "hi"]
        return [w for w in msgTokens if w not in stopWords]

    def filter_letters(self, msg):
        return ''.join([c for c in msg if c.isalpha() or c == ' '])

    def tokenize(self, msg):
        return msg.lower().split(' ')

    def for_coco(self, request_tokens):
        return "coco" in request_tokens


    def similarity_score(self, word_vector, word_vector2):
        """Calculating the mean Cosine Similarity of the most similar pairs from Possible pairs in product(vec1, vec2)

        Parameters
        ----------

        word_vector : list of str
            pythons' list of words

        word_vector2 : list of str
            pythons' list of words

        return : float
            mean cosine similarity of the most similar possible pairs in product(`vec1`, `vec2`)

        """
        max_scores = []
        # Initializing deep copy arrays, as pythons arrays are mutable.
        # Also, additional vectors are needed for duplicated words.
        vec1 = word_vector.copy()
        vec2 = word_vector2.copy()
        word_cuples = list(itertools.product(vec1, vec2))
        while word_cuples:
            scores = np.array(list(map(lambda tupl: self.word2vec.similarity(tupl[0], tupl[1]), word_cuples)))
            max_scores.append(scores.max())
            best_tupl = word_cuples[scores.argmax()]
            vec1.remove(best_tupl[0])
            vec2.remove(best_tupl[1])
            word_cuples = list(itertools.product(vec1, vec2))
        return(np.array(max_scores).mean())

if __name__ == '__main__':
    pass
    # Testing similarity_score
    # ta = TextAnalyzer.get_instance()
    # l1 = "mathematics is a hard subject please that a lot of people can't grasp".split(' ')
    # l2 = "software engineer is not easy as well and a lot of people are trying hard to learn it".split(' ')
    # l1 = ta.remove_stop_words(l1)
    # l2 = ta.remove_stop_words(l2)
    # print(ta.similarity_score(l1, l2))