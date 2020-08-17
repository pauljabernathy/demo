import unittest
from cleanser import *
import pandas as pd


class RawFightsCleanserTest(unittest.TestCase):

    def setUp(self):
        self.cleanser = RawFightsCleanser()

    def test_cleanse_column_names(self):
        df = pd.DataFrame(data=
                          {'r_fighter': ['John Wayne', 'Ghenghis Khan', 'Billy the Kid'],
                           'b_fighter': ['Darth Vader', 'Severus Snape', 'Gandalf'],
                           'R_SIG_STR.': ['1 of 10', '3 of 15', '8 of 9'],
                           'B_SIG_STR': ['4 of 5', '18 of 20', '6 of 7'],
                           'r_body.': ['1 of 5', '2 of 10', '4 of 12'],
                           'b_BODY': ['3 of 4', '9 of 10', '8 of 12']
                           })

        self.assertEqual({'r_fighter', 'b_fighter', 'R_SIG_STR.', 'B_SIG_STR', 'r_body.', 'b_BODY'},
                         set(df.columns))
        df = self.cleanser.cleanse_column_names(df)
        self.assertEqual({'r_fighter', 'b_fighter', 'r_sig_str', 'b_sig_str', 'r_body', 'b_body'},
                         set(df.columns))

    def test_expand_column_names(self):
        col1 = 'sig_str'
        col2 = 'body'

        set1 = self.cleanser.expand_column_names('sig_str')
        self.assertEqual(set(set1), {'r_sig_str_att', 'b_sig_str_att', 'r_sig_str_suc', 'b_sig_str_suc',
                                     'r_sig_str_pct', 'b_sig_str_pct',
                                     'r_sig_str_ratio', 'b_sig_str_ratio', 'sig_str_diff'})

        set2 = self.cleanser.expand_column_names('body')
        self.assertEqual(set(set2), {'r_body_att', 'b_body_att', 'r_body_suc', 'b_body_suc',
                                     'r_body_pct', 'b_body_pct',
                                     'r_body_ratio', 'b_body_ratio', 'body_diff'})

    def test_current_column_names(self):
        col1 = 'sig_str'
        col2 = 'body'
        sig_str_cols = self.cleanser.current_column_names(col1)
        self.assertEqual(['r_sig_str', 'b_sig_str'], sig_str_cols)

        body_cols = self.cleanser.current_column_names(col2)
        self.assertEqual(['r_body', 'b_body'], body_cols)


    def test_split_column_text(self):
        split1 = self.cleanser.split_column_text("73 of 150")
        #self.assertEqual(np.array([73, 150]), self.cleanser.split_column_text("73 of 150"))
        #self.assertEqual([4, 5], self.cleanser.split_column_text("4 of 5"))
        self.assertEqual(0, (np.array([73, 150]) != self.cleanser.split_column_text("73 of 150")).sum())
        self.assertEqual(0, (np.array([4, 5]) != self.cleanser.split_column_text("4 of 5")).sum())
        print(self.cleanser.split_column_text("0 of 0"))

    def test_split_column_text_tuple(self):
        split1 = self.cleanser.split_column_text_tuple("73 of 150")
        x = '29'

    def test_split_composite_columns(self):
        current_stems = self.cleanser.composite_column_stems
        test_stems = ['sig_str', 'body', 'td']
        self.cleanser.composite_column_stems = test_stems
        df = pd.DataFrame(data=
                          {'r_fighter': ['John Wayne', 'Ghenghis Khan', 'Billy the Kid', 'Aladdin'],
                           'b_fighter': ['Darth Vader', 'Severus Snape', 'Gandalf', 'Ali Babba'],
                           'r_sig_str': ['1 of 10', '3 of 15', '8 of 9', '0 of 2'],
                           'b_sig_str': ['4 of 5', '18 of 20', '6 of 7', '0 of 0'],
                           'r_sig_str_pct': ['10%', '20%', '89%', '0%'],
                           'b_sig_str_pct': ['80%', '90%', '86%', '0%'],
                           'r_body': ['1 of 5', '2 of 10', '4 of 12', '0 of 0'],
                           'r_body_pct': ['20%', '20%', '33%', '0%'],
                           'b_body': ['3 of 4', '9 of 10', '8 of 12', '0 of 1'],
                           'b_body_pct': ['75%', '90%', '67%', '0%'],
                           'r_td': ['0 of 0', '1 of 12', '3 of 4', '8 of 8'],
                           'b_td': ['20 of 20', '3 of 4', '9 of 9', '1 of 8']
                           })
        self.cleanser.split_composite_columns(df)
        column_names = list(df.columns)
        self.assertTrue('r_sig_str_att' in column_names)
        self.assertTrue('r_sig_str_suc' in column_names)
        self.assertTrue('r_sig_str_pct' in column_names)
        self.assertTrue('r_sig_str_ratio' in column_names)
        self.assertTrue('b_sig_str_att' in column_names)
        self.assertTrue('b_sig_str_suc' in column_names)
        self.assertTrue('b_sig_str_ratio' in column_names)

        self.assertTrue('r_body_att' in column_names)
        self.assertTrue('r_body_suc' in column_names)
        self.assertTrue('r_body_pct' in column_names)
        self.assertTrue('r_body_ratio' in column_names)
        self.assertTrue('b_body_att' in column_names)
        self.assertTrue('b_body_suc' in column_names)
        self.assertTrue('b_body_ratio' in column_names)

        self.assertTrue('r_td' in column_names)
        self.assertTrue('b_td' in column_names)
        self.assertTrue('r_td_att' in column_names)
        self.assertTrue('b_td_att' in column_names)
        self.assertTrue('r_td_suc' in column_names)
        self.assertTrue('b_td_suc' in column_names)
        self.assertTrue('r_td_ratio' in column_names)
        self.assertTrue('b_td_ratio' in column_names)
        self.assertTrue('r_td_pct' in column_names)
        self.assertTrue('b_td_pct' in column_names)
        self.assertTrue('td_diff' in column_names)

        print(df)
        self.assertEqual(df.iloc[3].r_sig_str_ratio, 0.0)
        self.assertEqual(df.iloc[3].b_sig_str_ratio, 0.0)
        self.assertEqual(df.iloc[3].r_body_ratio, 0.0)

        # some of the values but not all
        self.assertEqual(df.iloc[0].r_sig_str_ratio, 0.1)
        self.assertEqual(df.iloc[1].r_sig_str_ratio, 1/5)
        self.assertEqual(df.iloc[2].r_sig_str_ratio, 8/9)
        self.assertEqual(df.iloc[3].r_sig_str_ratio, 0)

        self.assertEqual(0, df.iloc[0].r_td_att)
        self.assertEqual(0, df.iloc[0].r_td_suc)
        self.assertEqual(0, df.iloc[0].r_td_ratio)
        self.assertEqual(0, df.iloc[0].r_td_pct)

        self.assertEqual(12, df.iloc[1].r_td_att)
        self.assertEqual(1, df.iloc[1].r_td_suc)
        self.assertEqual(1/12, df.iloc[1].r_td_ratio)
        self.assertEqual(8, df.iloc[1].r_td_pct)

        self.assertEqual(-6, df.iloc[2].td_diff)
        self.assertEqual(7, df.iloc[3].td_diff)


        self.cleanser.composite_column_stems = current_stems
        fights = pd.read_csv('raw_total_fight_data.csv', sep=';')
        fights = self.cleanser.cleanse_column_names(fights)
        print(fights.columns)
        #set(fights.columns)
        #before = set(fights.columns)
        before = set(fights.columns)
        fights = self.cleanser.split_composite_columns(fights)
        after = set(fights.columns)
        print(fights.columns)
        self.assertGreater(len(after.difference(before)), 0)
        x = 'breakpoint'

    def test_cleanse(self):
        fights = pd.read_csv('raw_total_fight_data.csv', sep=';')
        self.assertGreater(fights.isnull().sum().sum(), 0)
        fights = self.cleanser.cleanse(fights)
        self.assertEqual(fights.isnull().sum().sum(), 0)
        # Other aspects of cleansing are tested in other tests.


if __name__ == '__main__':
    unittest.main()
