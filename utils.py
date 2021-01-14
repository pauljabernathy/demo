import pandas as pd


def get_fights_for_fighter(fighter_name, fights_df):
    return fights_df[(fights_df.r_fighter == fighter_name) | (fights_df.b_fighter == fighter_name)]


class RecordSummary:
    def __init__(self, name, wins, losses, ties):
        self.name = name
        self.wins = wins
        self.losses = losses
        self.ties = ties

    def __str__(self):
        return '{ name: ' + self.name + ', ' + str(self.wins) + ', ' + str(self.losses) + ', ' + str(
            self.ties) + ', ' + str(self.wins / (self.wins + self.losses)) + '}'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return (self.wins < other.wins)


def get_record_summary(fighter_name, fights_df):
    summary = {}
    summary['wins'] = (fights_df['winner'] == fighter_name).sum()
    summary['losses'] = (fights_df['winner'] == fighter_name).sum()
    summary['losses'] = (fights_df['loser'] == fighter_name).sum()
    summary['ties'] = (((fights_df['r_fighter'] == fighter_name) | (fights_df['b_fighter'] == fighter_name)) & (
                fights_df['winner'] == 'None')).sum()
    summary['win loss ratio'] = summary['wins'] / (summary['wins'] + summary['losses'])

    return summary


def get_prior_record_summary(fighter_name, fights_df, date):
    summary = {}
    summary['wins'] = ((fights_df['winner'] == fighter_name) & (fights_df.date < date)).sum()
    summary['losses'] = ((fights_df['loser'] == fighter_name) & (fights_df.date < date)).sum()
    summary['ties'] = ((((fights_df['r_fighter'] == fighter_name) | (fights_df['b_fighter'] == fighter_name)) & (
            fights_df['winner'] == 'None')) & (fights_df.date < date)).sum()
    if summary['wins'] + summary['losses'] > 0:
        summary['win loss ratio'] = summary['wins'] / (summary['wins'] + summary['losses'])
    else:
        summary['win loss ratio'] = 0
    return summary


def linear_scale_column(column):
    column = pd.Series(column)  # make sure it is a series so we can do a vectorized operation
    min = column.min()
    max = column.max()
    #column = column.apply(lambda n: (n - max) / (max - min) + 1)
    column = column.apply(lambda n: (n - min) / (max - min))
    return column
