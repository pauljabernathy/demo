from analyzer_base import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class LogisticRegressionAnalyzer(AnalyzerBase):

    def __init__(self):
        super().__init__()

    def analyze(self):
        lr = LogisticRegression()
        lr.fit(self.x_train, self.y_train)
        lg_predictions = lr.predict(self.x_test)
        confusion_matrix(self.y_test, lg_predictions)
        print(accuracy_score(self.y_test, lg_predictions))
        print(classification_report(self.y_test, lg_predictions))


lga = LogisticRegressionAnalyzer()
lga.analyze()
