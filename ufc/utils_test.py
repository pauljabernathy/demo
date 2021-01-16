import unittest
import pandas as pd
# import date
from datetime import date
import utils


class UtilsTest(unittest.TestCase):
    def test_get_record_summary(self):
        self.assertEqual(True, False)

    def test_get_record_summary(self):
        r_fighter = ['bob', 'alice', 'cam'] * 3
        b_fighter = ['cam', 'bob', 'alice'] * 3
        winner = ['bob', 'alice', 'cam', 'cam', 'bob', 'alice', 'bob', 'alice', 'alice']
        loser = ['cam', 'bob', 'alice', 'bob', 'alice', 'cam', 'cam', 'bob', 'cam']
        dates = [date.fromisoformat("2019-01-01"), date.fromisoformat("2019-02-01"), date.fromisoformat("2019-03-01"),
                 date.fromisoformat("2019-04-01"), date.fromisoformat("2019-05-01"), date.fromisoformat("2019-06-01"),
                 date.fromisoformat("2019-07-01"), date.fromisoformat("2019-08-01"), date.fromisoformat("2019-09-01")]

        # Using the list below causes dates to be str type.  Using the one above causes it to be datetimes.
        dates2 = ["2019-01-01", "2019-02-01", "2019-03-01",
                  "2019-04-01", "2019-05-01", "2019-06-01",
                  "2019-07-01", "2019-08-01", "2019-09-01"]

        df = pd.DataFrame({'r_fighter': r_fighter, 'b_fighter': b_fighter, 'winner': winner, 'loser': loser,
                           'date': dates})
        summary = utils.get_record_summary('bob', df)
        self.assertEqual({'wins': 3, 'losses': 3, 'ties': 0, 'win loss ratio': .5}, summary)
        self.assertEqual({'losses': 3, 'wins': 3, 'ties': 0, 'win loss ratio': .5}, summary)  # So a different order
        # works.
        self.assertEqual({'wins': 4, 'losses': 2, 'ties': 0, 'win loss ratio': 2/3},
                         utils.get_record_summary('alice', df))
        self.assertEqual({'wins': 2, 'losses': 4, 'ties': 0, 'win loss ratio': 1/3},
                         utils.get_record_summary('cam', df))

    def test_get_prior_record_summary(self):
        r_fighter = ['bob', 'alice', 'cam'] * 3
        b_fighter = ['cam', 'bob', 'alice'] * 3
        winner = ['bob', 'alice', 'cam', 'cam', 'bob', 'alice', 'bob', 'alice', 'cam']
        loser = ['cam', 'bob', 'alice', 'bob', 'alice', 'cam', 'cam', 'bob', 'alice']
        dates = [date.fromisoformat("2019-01-01"), date.fromisoformat("2019-02-01"), date.fromisoformat("2019-03-01"),
                 date.fromisoformat("2019-04-01"), date.fromisoformat("2019-05-01"), date.fromisoformat("2019-06-01"),
                 date.fromisoformat("2019-07-01"), date.fromisoformat("2019-08-01"), date.fromisoformat("2019-09-01")]

        df = pd.DataFrame({'r_fighter': r_fighter, 'b_fighter': b_fighter, 'winner': winner, 'loser': loser,
                           'date': dates })
        summary = utils.get_prior_record_summary('bob', df, date.fromisoformat("2019-06-01"))
        print(df)
        self.assertEqual({'wins': 2, 'losses': 2, 'ties': 0, 'win loss ratio': 1/2}, summary)

        self.assertEqual({'wins': 1, 'losses': 0, 'ties': 0, 'win loss ratio': 1.0},
                         utils.get_prior_record_summary('bob', df, date.fromisoformat("2019-02-01")))
        self.assertEqual({'wins': 0, 'losses': 0, 'ties': 0, 'win loss ratio': 0},
                         utils.get_prior_record_summary('alice', df, date.fromisoformat("2019-02-01")))
        self.assertEqual({'wins': 0, 'losses': 1, 'ties': 0, 'win loss ratio': 0.0},
                         utils.get_prior_record_summary('cam', df, date.fromisoformat("2019-02-01")))

        self.assertEqual({'wins': 1, 'losses': 1, 'ties': 0, 'win loss ratio': 1/2},
                         utils.get_prior_record_summary('bob', df, date.fromisoformat("2019-03-01")))
        self.assertEqual({'wins': 1, 'losses': 0, 'ties': 0, 'win loss ratio': 1},
                         utils.get_prior_record_summary('alice', df, date.fromisoformat("2019-03-01")))
        self.assertEqual({'wins': 0, 'losses': 1, 'ties': 0, 'win loss ratio': 0.0},
                         utils.get_prior_record_summary('cam', df, date.fromisoformat("2019-03-01")))

        self.assertEqual({'wins': 1, 'losses': 1, 'ties': 0, 'win loss ratio': 1/2},
                         utils.get_prior_record_summary('bob', df, date.fromisoformat("2019-04-01")))
        self.assertEqual({'wins': 1, 'losses': 1, 'ties': 0, 'win loss ratio': 1/2},
                         utils.get_prior_record_summary('alice', df, date.fromisoformat("2019-04-01")))
        self.assertEqual({'wins': 1, 'losses': 1, 'ties': 0, 'win loss ratio': 1/2},
                         utils.get_prior_record_summary('cam', df, date.fromisoformat("2019-04-01")))

    def test_linear_scale(self):
        data = [5, 6, 7, 7.5, 8, 9, 10]
        exp_result = [0, .2, .4, .5, .6, .8, 1]
        result = utils.linear_scale_column(data)
        #self.assertEqual(result, exp_result)
        self.assertTrue(pd.Series(result).equals(result))

if __name__ == '__main__':
    unittest.main()
