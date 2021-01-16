import pandas as pd
import numpy as np

# TODO: Make this not a class, but a module for utilities functions.
# TODO: "Cleanser" might not be the best name.

ATTEMPT = '_att'
SUCCESS = '_suc'
PERCENT = '_pct'
RATIO = '_ratio'
DIFFERENCE = '_diff'


class RawFightsCleanser:

    composite_column_stems = ['sig_str', 'total_str', 'td', 'head', 'body', 'leg', 'distance', 'clinch', 'ground']

    def cleanse_column_names(self, fights):
        """
        Make the column names easier to work with - all lower case (easier to type) and no . at the end.
        :param fights: a DataFrame for the raw_total_fights_data.  Othere dfs coule work too, but this function is
        designed for a couple of things I saw in that specific data set.
        :return: the same fights DataFrame, but with some columns renamed
        """
        # First, deal with columns with a period at the end.  Why is there a period at the end?  Not sure.
        cols_with_period = []
        new_column_names = {}
        for c in fights.columns:  # columns_of_interest:
            if c.endswith('.'):
                cols_with_period.append(c)
                new_column_names[c] = c.replace('.', '')
        fights = fights.rename(columns=new_column_names)

        # Now, make tham all lower case.
        col_names_map = {}
        for column in fights.columns:
            col_names_map[column] = column.lower()
        # col_names_map

        fights = fights.rename(columns=col_names_map)
        return fights

    # TODO: constants for column prefixes
    def expand_column_names(self, column_root):
        """
        Create column names from one of the x of y columns.  For example, the r_sig_str columns - should have
        r_sig_str_att, r_sig_str_suc, and r_sig_str_ratio.  Same for b.
        Some column stems have a _pct already; some do not.  split_composite_columns() will handle the _pct columns.
        _ratio and _pct are mostly redundant.  _pct is the integeter percent.  For example, a ratio of .47889 would
        become a _pct of 48.  This was because the ratio might be easier to use with a numeric algorithm,
        while the percent (being an int) is easier to put into a histogram graph.
        :param column_root:
        :return:
        """
        r = 'r_'
        b = 'b_'
        att = ATTEMPT
        suc = SUCCESS
        ratio = RATIO
        pct = PERCENT
        diff = DIFFERENCE
        return [r + column_root + att, b + column_root + att, r + column_root + suc, b + column_root + suc,
                r + column_root + ratio, b + column_root + ratio, r + column_root + pct, b + column_root + pct,
                column_root + diff]

    def current_column_names(self, column_root):
        return ['r_' + column_root, 'b_' + column_root]

    def split_column_text(self, text):
        string_values = text.split(' of ')
        return np.array([int(string_values[0]), int(string_values[1])])

    def split_column_text_tuple(self, text):
        string_values = text.split(' of ')
        return int(string_values[0]), int(string_values[1])

    def split_one_composite_column(self, column, df):
        pass

    def split_composite_columns(self, df):
        for stem in self.composite_column_stems:
            expanded = self.expand_column_names(stem)
            current = self.current_column_names(stem)
            r_suc_att_values = df[current[0]].apply(lambda x: self.split_column_text(x))
            b_suc_att_values = df[current[1]].apply(lambda x: self.split_column_text(x))
            r_suc = r_suc_att_values.apply(lambda v: v[0])
            r_att = r_suc_att_values.apply(lambda v: v[1])
            b_suc = b_suc_att_values.apply(lambda v: v[0])
            b_att = b_suc_att_values.apply(lambda v: v[1])

            r_ratio_value = r_suc / r_att #r_suc_att_values[0] / r_suc_att_values[1]
            b_ratio_value = b_suc / b_att  # b_suc_att_values[0] / b_suc_att_values[1]

            # Now, add the new columns
            df[expanded[0]] = r_att
            df[expanded[1]] = b_att
            df[expanded[2]] = r_suc
            df[expanded[3]] = b_suc
            df[expanded[4]] = r_ratio_value.replace(np.NaN, 0.0)  # If it was "0 of 0", make the ratio 0, not np.NaN
            df[expanded[5]] = b_ratio_value.replace(np.NaN, 0.0)

            # Where in an existing _pct column, remove the '%' and make it an int, not a string.
            # When there is not an existing _pct column, create one.
            r_percent_col = f'r_{stem}{PERCENT}'
            if r_percent_col in df.columns:
                df[r_percent_col] = df[r_percent_col].str.replace('%', '').apply(lambda x: int(x))
            else:
                df[expanded[6]] = (100 * df[expanded[4]]).apply(lambda n: int(n))

            b_percent_col = f'b_{stem}{PERCENT}'
            if b_percent_col in df.columns:
                df[b_percent_col] = df[b_percent_col].str.replace('%', '').apply(lambda x: int(x))
            else:
                df[expanded[7]] = (100 * df[expanded[5]]).apply(lambda n: int(n))

            # TODO:  Is there a way of keeping track of columns that does not involve matching numbers?  This is
            #  annoying.
            df[expanded[8]] = r_suc - b_suc

            # Delete the old columns
            # df.drop(current[0], axis=1)
            # df.drop(current[1], axis=1)

        # Return is not technically necessary, but I always prefer returning something from almost any function.
        return df

    def find_loser(self, row):
        if row.winner == row.r_fighter:
            return row.b_fighter
        elif row.winner == row.b_fighter:
            return row.r_fighter
        else:
            return 'None'

    def winner_b_r(self, row):
        if row['winner'] == row.r_fighter:
            return 'r'
        elif row['winner'] == row.b_fighter:
            return 'b'
        else:
            return 'None'

    def cleanse(self, fights):
        """
        Calls the cleansing functions in one function call.
        :param fights:
        :return:
        """
        fights = self.cleanse_column_names(fights)
        fights = self.split_composite_columns(fights)
        fights = fights.replace(np.NaN, 'None')
        fights['loser'] = fights.apply(self.find_loser, axis=1)
        fights['r_b_winner'] = fights.apply(lambda row: self.winner_b_r(row), axis=1)
        return fights

    def load_and_cleanse(self, filename, sep):
        """
        Combines loading the file and cleansing.
        :param filename:
        :param sep:
        :return:
        """
        fights = pd.read_csv(filename, sep=sep)
        fights = self.cleanse(fights)
        return fights

