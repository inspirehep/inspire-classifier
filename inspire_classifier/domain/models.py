
class CoreClassifier(object):
    def __init__(self, title, abstract):
        self.score_a = None
        self.score_b = None
        self.score_c = None

    def classify(self):
        self.score_a = 5
        self.score_b = 10
        self.score_c = 15
