import unittest
import fighters_cleanser
import pandas as pd
import constants


class FightersCleanserTest(unittest.TestCase):

    def test_convert_length_to_inches(self):
        # a couple of ways something could be missing
        #self.assertEqual(fighters_cleanser.convert_length_to_inches("0'0\""), 0)
        #self.assertEqual(fighters_cleanser.convert_length_to_inches("5'"), 60)

        # inches only, like in reach
        # self.assertEqual(fighters_cleanser.convert_length_to_inches("4\""), 64)
        # nevermind; not handling reach with this function

        # fight and inches, like in height
        self.assertEqual(fighters_cleanser.convert_length_to_inches("5'4\""), 64)
        self.assertEqual(fighters_cleanser.convert_length_to_inches("5'5\""), 65)
        self.assertEqual(fighters_cleanser.convert_length_to_inches("6'0\""), 72)
        self.assertEqual(fighters_cleanser.convert_length_to_inches("6'4\""), 76)
        self.assertEqual(fighters_cleanser.convert_length_to_inches("6'6\""), 78)

        # with a space
        self.assertEqual(fighters_cleanser.convert_length_to_inches("5' 4\""), 64)
        self.assertEqual(fighters_cleanser.convert_length_to_inches(" 5' 4\" "), 64)

    def test_convert_height(self):
        df = pd.DataFrame({'name': ['alice', 'bob', 'charlie', 'dick', 'everett'],
                           'height': ["5'4\"", "5' 5\"", "6'0\"", "6'4\"", "6'6\""]})
        df = fighters_cleanser.convert_heights(df, ['height'])
        # These are all equivalent for checking that the height values are [64, 65, 72, 76, 78]
        self.assertEqual([64, 65, 72, 76, 78], list(df.height))
        # You have to convert to a list to directly compare them like above
        # Or you can use .all() like below
        self.assertTrue(([64, 65, 72, 76, 78] == df.height).all())
        self.assertTrue(([64, 65, 72, 76, 78] == df.height).sum() == ([64, 65, 72, 76, 78] != df.height).count())

    def test_convert_reach(self):
        df = pd.DataFrame({'name': ['alice', 'bob', 'charlie', 'dick', 'everett'],
                           'reach_column_1': ['64"', '68"', '63"', '66"', '59"'],
                           'reach_column_2': ['1"', '2"', '3"', '4"', '17"']})
        df = fighters_cleanser.convert_reach(df, ['reach_column_1', 'reach_column_2'])
        # pandas trivia - the below two are equivalent
        self.assertTrue(([64, 68, 63, 66, 59] == df.reach_column_1).all())
        self.assertTrue((pd.Series([64, 68, 63, 66, 59]).equals(df.reach_column_1)))
        self.assertTrue(([1, 2, 3, 4, 17] == df.reach_column_2).all())

    def test_convert_weight(self):
        df = pd.DataFrame({'r_fighter': ['r1', 'r2', 'r3'], 'blue_fighter': ['b1', 'b2', 'b3'],
                           'r_weight': ['135 lbs.', '187', '548 lbs..'],
                           'b_weight': ['30', '57 lbs', '165 lbs.']
                           })
        df = fighters_cleanser.convert_weight(df, ['b_weight', 'r_weight'])
        self.assertTrue(([135, 187, 548] == df.r_weight).all())
        self.assertTrue(([30, 57, 165] == df.b_weight).all())

    def test_convert_dates(self):
        df = pd.DataFrame({'r_fighter': ['r1', 'r2', 'r3'], 'blue_fighter': ['b1', 'b2', 'b3'],
                           'r_weight': ['135 lbs.', '187', '548 lbs..'],
                           'b_weight': ['30', '57 lbs', '165 lbs.'],
                           'r_dob': ['Aug 08, 1988', 'Sep 28, 1976', 'Dec 08, 1981'],
                           'b_dob': ["Jun 26, 1985", "Apr 30, 1988", "Mar 07, 1989"]
                           })
        df = fighters_cleanser.convert_dates(df, ['r_dob', 'b_dob'])
        self.assertTrue(df.r_dob.dtype == 'datetime64[ns]')
        self.assertTrue(df.b_dob.dtype == 'datetime64[ns]')

    def test_add_body_diffs(self):
        df = pd.DataFrame({'r_fighter': ['r1', 'r2', 'r3'], 'blue_fighter': ['b1', 'b2', 'b3'],
                           'r_height': [64, 68, 63],
                           'b_height': [66, 66, 66],
                           'r_weight': [135, 187, 148],
                           'b_weight': [30, 57, 165],
                           'r_reach': [64, 68, 63],
                           'b_reach': [71, 62, 58]
                           })
        df = fighters_cleanser.add_body_diffs(df)
        self.assertTrue(([-2, 2, -3] == df.height_diff).all())
        self.assertTrue(([105, 130, -17] == df.weight_diff).all())
        self.assertTrue(([-7, 6, 5] == df.reach_diff).all())

    def test_scale_column(self):
        a = [1, 2, 3, 4, 5]
        scaled_a = fighters_cleanser.scale_column(a)
        # using pd.Series().std()
        exp_scaled_a = [-1.2649110640673518, -0.6324555320336759, 0.0, 0.6324555320336759, 1.2649110640673518]
        # using np.std()
        exp_scaled_a = [-1.414213562373095, -0.7071067811865475, 0.0, 0.7071067811865475, 1.414213562373095]
        self.assertTrue((scaled_a == pd.Series(exp_scaled_a)).all())
        self.assertTrue(scaled_a.equals(pd.Series(exp_scaled_a)))

    def test_scale_columns(self):
        df = pd.DataFrame({'r_fighter': ['r1', 'r2', 'r3'], 'blue_fighter': ['b1', 'b2', 'b3'],
                          'a': [2, 3, 4], 'r_height': [64, 68, 63], 'b_height': [66, 66, 66],
        })
        df = fighters_cleanser.scale_columns(df, ['a', 'r_height'])
        # using pandas std()
        exp_r_height = [-0.3779644730092272, 0.0, -0.7559289460184544]
        # using numpy std()
        exp_a = [-1.224744871391589, 0.0,  1.224744871391589]
        exp_r_height = [-0.4629100498862757, 1.3887301496588271, -0.9258200997725514]
        exp_df = pd.DataFrame({'r_fighter': ['r1', 'r2', 'r3'], 'blue_fighter': ['b1', 'b2', 'b3'],
            'a': exp_a, 'r_height': exp_r_height, 'b_height': [66, 66, 66]
        })
        self.assertTrue(exp_df.equals(df))

    def test_load_and_cleanse(self):
        fighters = fighters_cleanser.load_and_cleanse(constants.DEFAULT_FIGHTERS_FILE_NAME)
        self.assertTrue((['fighter_name', 'height', 'weight', 'reach', 'stance', 'dob'] == fighters.columns).all())
        self.assertTrue(fighters.height.dtype == 'int64')
        self.assertTrue(fighters.reach.dtype == 'int64')
        self.assertTrue(fighters.weight.dtype == 'int64')
        self.assertTrue(fighters.dob.dtype == 'datetime64[ns]')

    def test_load_cleanse_and_merge(self):
        combined = fighters_cleanser.load_cleanse_and_merge(constants.DEFAULT_FIGHTERS_FILE_NAME,
                                                            constants.DEFAULT_FIGHTS_FILE_NAME)
        some_columns_that_should_be_there = ['r_height', 'r_weight', 'r_reach', 'b_height', 'b_weight', 'b_reach',
                                        'r_age', 'b_age', 'age_diff', 'height_diff', 'weight_diff', 'reach_diff']
        for column in some_columns_that_should_be_there:
            self.assertTrue(column in combined.columns)

        self.assertTrue(combined.r_height.dtype == 'int64')
        self.assertTrue(combined.r_reach.dtype == 'int64')
        self.assertTrue(combined.r_weight.dtype == 'int64')
        self.assertTrue(combined.b_height.dtype == 'int64')
        self.assertTrue(combined.b_reach.dtype == 'int64')
        self.assertTrue(combined.b_weight.dtype == 'int64')
        self.assertTrue(combined.date.dtype == 'datetime64[ns]')
        self.assertTrue(combined.r_dob.dtype == 'datetime64[ns]')
        self.assertTrue(combined.b_dob.dtype == 'datetime64[ns]')
        self.assertTrue(combined.r_age.dtype == 'float64')
        self.assertTrue(combined.b_age.dtype == 'float64')
        self.assertTrue(combined.age_diff.dtype == 'float64')


if __name__ == '__main__':
    unittest.main()
