import pandas as pd
import numpy as np
from cleanser import RawFightsCleanser
from dateutil import parser
from datetime import datetime
import re
import constants
import utils


# TODO: Decide if this and the other cleanser really should be classes, since the functions are all in effect static.
# Let's try it with being a class.
#class FightersCleanser:

 #   def __init_(self):
 #       self.fights_cleanser = RawFightsCleanser()
fights_cleanser = RawFightsCleanser()

def cleanse_column_names(fighters):
    # Since RawFightsCleanser.cleanse_column_names does the same thing we want to do here, we can reuse it
    # TODO: see if you can make a generic cleanser class that both of these inherit from
    return fights_cleanser.cleanse_column_names(fighters)


def get_column_name_replacements(cols_to_rename, prefix):
    column_name_replacements = {}
    for column in cols_to_rename:
        column_name_replacements[column] = prefix + column
    return column_name_replacements


def join_with_fights(fighters, fights):
    cols_to_rename = list(fighters.columns[1:])
    combined = pd.merge(fights, fighters, left_on='r_fighter', right_on='fighter_name')
    combined = combined.drop('fighter_name', axis=1)
    combined = combined.rename(columns=get_column_name_replacements(cols_to_rename, 'r_'))
    combined = pd.merge(combined, fighters, left_on='b_fighter', right_on='fighter_name')
    combined = combined.drop('fighter_name', axis=1)
    combined = combined.rename(columns=get_column_name_replacements(cols_to_rename, 'b_'))
    return combined


def convert_length_to_inches(length_as_string):
    """
    converts a length in the format of feet' inches", such as 5'4", to an integer number of inches
    :param length_as_string: a string like 5'4"
    :return: a integer, like 64
    """
    numbers = re.split("'", length_as_string)
    for i in range(len(numbers)):
        numbers[i] = numbers[i].replace('"', '')
        if numbers[i] == '':
            numbers[i] = 0
        numbers[i] = int(numbers[i])
    return numbers[0] * 12 + numbers[1]


def convert_heights(df, columns):
    """
    convert height from something like 6'1" to something like 73
    :param df:
    :param columns:
    :return:
    """
    for column in columns:
        # df[column] = df[column].apply(convert_length_to_inches)
        df[column] = df[column].apply(lambda length: convert_length_to_inches(length))
    return df


def convert_reach(df, columns):
    """
    convert reach from something like 64" to 64; not a complicated function
    :param df:
    :param columns:
    :return: reach as an integer
    """
    for column in columns:
        df[column] = df[column].apply(lambda reach_str: int(reach_str.replace('"', '')))
    return df


def convert_weight(df, columns):
    """
    removed the "lbs." from the weight column
    :param df:
    :param columns:
    :return: weight as an integer
    """
    for column in columns:
        df[column] = df[column].apply(lambda weight_str: int(weight_str.replace('lbs', '').replace('.', '')))
    return df


def convert_dates(df, columns):
    """
    change the date from a string to an object
    :param df:
    :param columns:
    :return:
    """
    for column in columns:
        df[column] = df[column].apply(lambda date_str: parser.parse(date_str))
    return df


def add_ages(combined):
    combined['r_age'] = (combined.date - combined.r_dob).apply(lambda t: t.days / 365)
    combined['b_age'] = (combined.date - combined.b_dob).apply(lambda t: t.days / 365)
    combined['age_diff'] = combined.r_age - combined.b_age
    return combined


def add_body_diffs(df):
    df['height_diff'] = df.r_height - df.b_height
    df['weight_diff'] = df.r_weight - df.b_weight
    df['reach_diff'] = df.r_reach - df.b_reach
    return df


# TODO: ability to scale by standard deviation or by the other linear scaling method
def scale_column(column):
    """
    scales the values in the column, currently by mean and standard deviation; that is, for each value, it converts
    it into how many standard deviations it is from the mean
    :param column: a pandas Series or list; all values must be numeric and it should be (at least approximately)
    normally
    distributed or it won't make sense
    :return: the series scaled
    """
    column = pd.Series(column)  # make sure it is a series so we can do a vectorized operation
    mean = column.mean()
    std = column.std()
    std = np.std(column)    # np divides by n, pandas divides by n - 1; we'll use np since we have the whole
    # population, not a sample
    column = column.apply(lambda n: (n - mean) / std)
    return column


def scale_columns(df, columns):
    for column in columns:
        df[column] = scale_column(df[column])
    return df


def add_records_for_row(row, fights):
    r_records = utils.get_prior_record_summary(row.r_fighter, fights, row.date)
    b_records = utils.get_prior_record_summary(row.b_fighter, fights, row.date)
    row.r_prior_wins = r_records['wins']
    row.r_prior_losses = r_records['losses']
    row.r_prior_ties = r_records['ties']
    row.b_prior_wins = b_records['wins']
    row.b_prior_losses = b_records['losses']
    row.b_prior_ties = b_records['ties']
    return row


def add_fictitious_records(fights, prior_wins=constants.DEFAULT_PRIOR_WINS, prior_losses=constants.DEFAULT_PRIOR_LOSSES,
                           prior_ties=constants.DEFAULTPRIOR_TIES):
    fights['r_prior_wins'] = 0
    fights['r_prior_losses'] = 0
    fights['r_prior_ties'] = 0
    fights['b_prior_wins'] = 0
    fights['b_prior_losses'] = 0
    fights['b_prior_ties'] = 0
    fights = fights.apply(lambda row: add_records_for_row(row, fights), axis=1)
    fights['r_prior_wins'] += prior_wins
    fights['b_prior_wins'] += prior_wins
    fights.r_prior_losses += prior_losses
    fights.b_prior_losses += prior_losses
    fights.r_prior_ties += prior_ties
    fights.b_prior_ties += prior_ties
    return fights


def find_win_loss_tie_pct(fights):
    fights['r_total_fights'] = fights.r_prior_wins + fights.r_prior_losses + fights.r_prior_ties
    fights['b_total_fights'] = fights.b_prior_wins + fights.b_prior_losses + fights.b_prior_ties
    fights['r_win_pct'] = fights.r_prior_wins / fights.r_total_fights
    fights['r_loss_pct'] = fights.r_prior_losses / fights.r_total_fights
    fights['r_tie_pct'] = fights.r_prior_ties / fights.r_total_fights

    fights['b_win_pct'] = fights.b_prior_wins / fights.b_total_fights
    fights['b_loss_pct'] = fights.b_prior_losses / fights.b_total_fights
    fights['b_tie_pct'] = fights.b_prior_ties / fights.b_total_fights
    return fights


def load_and_cleanse(fighters_file_name):
    """
    load the fighters dataset and cleanse
    :param fighters_file_name:
    :return:
    """
    fighters = pd.read_csv(fighters_file_name)
    fighters = cleanse_column_names(fighters)

    # If you cleanse the numeric data here, you don't have to deal 'r_height' and 'b_height' so you only do one column.
    fighters = fighters[fighters.height.isnull() == False]
    fighters = convert_heights(fighters, ['height'])

    fighters = fighters[fighters.reach.isnull() == False]
    fighters = convert_reach(fighters, ['reach'])

    fighters = fighters[fighters.weight.isnull() == False]
    fighters = convert_weight(fighters, ['weight'])

    fighters = fighters[fighters.dob.isnull() == False]
    fighters = convert_dates(fighters, ['dob'])
    return fighters


def load_cleanse_and_merge(fighters_file_name, fights_file_name):
    """
    load both data sets, cleanse, and merge them so that we have a record of the fights and stats about each fighter
    :param fighters_file_name:
    :param fights_file_name:
    :return:
    """

    fights = fights_cleanser.load_and_cleanse(fights_file_name, sep=';')
    fighters = load_and_cleanse(fighters_file_name)
    combined = join_with_fights(fighters, fights)
    combined = convert_dates(combined, ['date'])
    combined = add_ages(combined)
    combined = add_body_diffs(combined)
    # combined = add_fictitious_records(combined)
    return combined


if __name__ == '__main__':
    load_cleanse_and_merge(constants.DEFAULT_FIGHTERS_FILE_NAME, constants.DEFAULT_FIGHTS_FILE_NAME)
