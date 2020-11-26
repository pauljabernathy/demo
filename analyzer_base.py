from cleanser import RawFightsCleanser
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class AnalyzerBase:

    def __init__(self):
        self.cleanser = RawFightsCleanser()
        self.load_data()

    def load_data(self):
        self.fights = self.cleanser.load_and_cleanse('raw_total_fight_data.csv', sep=';')
        diff_columns = [c for c in self.fights.columns if '_diff' in c]
        diffs = self.fights[diff_columns + ['r_b_winner']]
        self.scaled_diffs = diffs.copy()
        for column in diff_columns:
            # col = scaled_diffs[column]
            mean = self.scaled_diffs[column].mean()
            sd = self.scaled_diffs[column].std()
            self.scaled_diffs[column] = self.scaled_diffs[column].apply(lambda x: (x - mean) / sd)
        self.scaled_diffs['r_won'] = self.scaled_diffs.r_b_winner.apply(lambda x: 1 if x == 'r' else 0)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.scaled_diffs[['sig_str_diff', 'total_str_diff', 'td_diff', 'head_diff', 'body_diff',
                          'leg_diff', 'distance_diff', 'clinch_diff', 'ground_diff']], self.scaled_diffs['r_won'],
            test_size=.33, random_state=1)