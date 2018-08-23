
class CoreClassifier(object):
    def __init__(self):
        self.score_a = None
        self.score_b = None
        self.score_c = None

    def classify(self, title, abstract):
        self.score_a = 0.1
        self.score_b = 0.2
        self.score_c = 0.7
