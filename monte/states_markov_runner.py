
import rooms_markov_chain as rmc
from prob_dist import ProbDist
from scipy.stats import chisquare
import pandas as pd
import numpy as np
from pathlib import Path
import constants
import rooms_probs

original_results_columns = []
default_columns = ["num_chains", "chain_length", "starting_state"]
results_columns = ["num_chains", "chain_length", "all_A_B", "all_A_C", "all_B_C", "end_A_B",
                   "end_A_C", "end_B_C"]
results_file_name = "sim_results/all_results.csv"
chi_square_results_file_name = "sim_results/chi_square_results.csv"


def get_existing_results(file_name):
    the_file = Path(file_name)
    if the_file.exists():
        existing_results = pd.read_csv(file_name)
    else:
        # create a new df with just the three parameter columns
        existing_results = pd.DataFrame(columns=default_columns)
    return existing_results


def once_from_each_starting_state(num_chains, chain_length, dist_map, callback_function):
    for state in dist_map:
        all_states_hist, ending_states_hist = rmc.do_sim(dist_map, num_chains=num_chains, chain_length=chain_length,
                                                         display_each=False, starting_state=state, display_summary=False)
        if callback_function is not None:
            callback_function(all_states_hist, ending_states_hist, state)


def save_results(all_states_hist, ending_states_hist, starting_state, file_name):
    # The indexes of the two histograms need to match.  Theoretically, they always will.
    #assert all_states_hist.index == ending_states_hist.index
    print(all_states_hist)
    print(ending_states_hist)
    if set(all_states_hist.index) != set(ending_states_hist.index):
        return
        # Trying this instead of the assert below so that one of many groups of chains won't cause the whole thing to
        # crash.
    #assert set(all_states_hist.index) == set(ending_states_hist.index)
    # rmc.do_sim() sorts the histograms by values.  For simulations with many chains and an long chain length,
    # the sort will generally be the same from one run to the next and between the all and end histograms.  But that is
    # not guaranteed.  So sort them both alphabetically to force them to all have a matching index.
    all_states_hist = all_states_hist.sort_index()
    ending_states_hist = ending_states_hist.sort_index()
    assert all_states_hist.index.equals(ending_states_hist.index)

    all_columns = pd.Series(all_states_hist.index) + "_all"
    end_columns = pd.Series(ending_states_hist.index) + "_end"
    all_columns = default_columns + list(all_columns) + list(end_columns)
    existing_results = get_existing_results(file_name)

    if not existing_results.empty:
        # The columns of the existing data and the new data should match.  If not, it won't work.
        assert all_columns == list(existing_results.columns)

    new_results = pd.DataFrame()#columns=all_columns)
    num_chains = ending_states_hist['values'].sum()
    new_results['num_chains'] = [num_chains]
    new_results['chain_length'] = [all_states_hist['values'].sum() / num_chains]
    new_results['starting_state'] = [starting_state]

    # Now build the new rows that will be added to the existing_results df.
    for index, row in all_states_hist.iterrows():
        new_results[index + '_all'] = row['values']

    for index, row in ending_states_hist.iterrows():
        new_results[index + '_end'] = row['values']

    existing_results = existing_results.append(new_results)

    # TODO:  I guess this is a bug...assuming it will be a csv file
    existing_results.to_csv(file_name, index=False)


def save_results_3_states(all_states_hist, ending_states_hist, starting_state):
    save_results(all_states_hist, ending_states_hist, starting_state, "sim_results/3_states_results.csv")


def get_letter_dist_map():
    A = 'A'
    B = 'B'
    C = 'C'

    a_probs = ProbDist({B: 7, C: 3}, id='A')
    b_probs = ProbDist({A: 7, C: 3}, id='B')
    c_probs = ProbDist({A: 5, B: 5}, id='C')

    letter_dist_map = {
        A: a_probs,
        B: b_probs,
        C: c_probs
    }

    return letter_dist_map

def get_nine_rooms_dist_map():
    PANTRY = 'pantry'
    KITCHEN = 'kitchen'
    SCHOOL_ROOM = 'school'
    DEN = 'den'
    ENTRY_HALL = 'entry'
    OFFICE = 'office'
    HALL = 'hall'
    BEDROOM = 'bedroom'
    BATHROOM = 'bathroom'
    kitchen_probs = ProbDist({SCHOOL_ROOM: 1, PANTRY: 3, DEN: 6}, id=KITCHEN)
    pantry_probs = ProbDist({KITCHEN: 1}, id=PANTRY)
    school_room_probs = ProbDist({KITCHEN: 2, OFFICE: 1}, id=SCHOOL_ROOM)
    office_probs = ProbDist({SCHOOL_ROOM: 2, ENTRY_HALL: 3}, id=OFFICE)
    entry_hall_probs = ProbDist({OFFICE: 1, DEN: 1}, id=ENTRY_HALL)
    den_probs = ProbDist({KITCHEN: 4, ENTRY_HALL: 2, HALL: 4}, id=DEN)
    hall_probs = ProbDist({BEDROOM: 5, DEN: 5}, id=HALL)
    bedroom_probs = ProbDist({HALL: 6, BATHROOM: 4}, id=BEDROOM)
    bathroom_probs = ProbDist({BEDROOM: 1}, id=BATHROOM)

    pantry_probs = ProbDist({KITCHEN: constants.kp}, id=PANTRY)
    kitchen_probs = ProbDist({SCHOOL_ROOM: constants.sk, PANTRY: constants.pk, DEN: constants.dk}, id=KITCHEN)
    school_room_probs = ProbDist({KITCHEN: constants.ks, OFFICE: constants.fs}, id=SCHOOL_ROOM)
    office_probs = ProbDist({SCHOOL_ROOM: constants.sf, ENTRY_HALL: constants.ef}, id=OFFICE)
    entry_hall_probs = ProbDist({OFFICE: constants.fe, DEN: constants.de}, id=ENTRY_HALL)
    den_probs = ProbDist({KITCHEN: constants.kd, ENTRY_HALL: constants.ed, HALL: constants.hd}, id=DEN)
    hall_probs = ProbDist({BEDROOM: constants.rh, DEN: constants.dh}, id=HALL)
    bedroom_probs = ProbDist({HALL: constants.hr, BATHROOM: constants.tr}, id=BEDROOM)
    bathroom_probs = ProbDist({BEDROOM: constants.rt}, id=BATHROOM)

    house_dist_map = {
        KITCHEN: kitchen_probs,
        PANTRY: pantry_probs,
        SCHOOL_ROOM: school_room_probs,
        OFFICE: office_probs,
        ENTRY_HALL: entry_hall_probs,
        DEN: den_probs,
        HALL: hall_probs,
        BEDROOM: bedroom_probs,
        BATHROOM: bathroom_probs
    }
    return house_dist_map

def test_run():
    dist_map = get_letter_dist_map()
    num_cycles = 10
    for i in range(num_cycles):
        once_from_each_starting_state(10, 100000, dist_map, save_results_3_states)


def get_stats_for_params_3_state(df, num_chains, chain_length, exp_ratios):
    df_matching_params = df[(df.num_chains == num_chains) & (df.chain_length == chain_length)]
    abc = df_matching_params[['starting_state', 'A_all', 'B_all', 'C_all', 'A_end', 'B_end', 'C_end']].groupby('starting_state').sum()
    exp_numbers = [num_chains * chain_length * df_matching_params.shape[0] * r for r in exp_ratios] + \
                  [num_chains * df_matching_params.shape[0] * r for r in exp_ratios]

    # TODO: don't hardcode for the three state problem
    print(abc)

    result = {}
    result["A all, exp"] = chisquare(abc.loc['A'][['A_all', 'B_all', 'C_all']], np.array(exp_numbers[0:3]) / 3).pvalue
    result["B all, exp"] = chisquare(abc.loc['B'][['A_all', 'B_all', 'C_all']], np.array(exp_numbers[0:3]) / 3).pvalue
    result["C all, exp"] = chisquare(abc.loc['C'][['A_all', 'B_all', 'C_all']], np.array(exp_numbers[0:3]) / 3).pvalue
    result["A all, B all"] = chisquare(abc.loc['A'][['A_all', 'B_all', 'C_all']], abc.loc['B'][['A_all', 'B_all',
                                                                                                'C_all']]).pvalue
    result["A all, C all"] = chisquare(abc.loc['A'][['A_all', 'B_all', 'C_all']], abc.loc['C'][['A_all', 'B_all',
                                                                                              'C_all']]).pvalue
    result["B all, C all"] = chisquare(abc.loc['B'][['A_all', 'B_all', 'C_all']], abc.loc['C'][['A_all', 'B_all',
                                                                                              'C_all']]).pvalue
    result["A end, B end"] = chisquare(abc.loc['A'][['A_end', 'B_end', 'C_end']], abc.loc['B'][['A_end', 'B_end',
                                                                                     'C_end']]).pvalue
    result["A end, C end"] = chisquare(abc.loc['A'][['A_end', 'B_end', 'C_end']], abc.loc['C'][['A_end', 'B_end',
                                                                                     'C_end']]).pvalue
    result["B end, C end"] = chisquare(abc.loc['B'][['A_end', 'B_end', 'C_end']], abc.loc['C'][['A_end', 'B_end',
                                                                                     'C_end']]).pvalue
    #result[] =
    #result[] =
    return result


def get_stats_for_params(results_df, num_chains, chain_length, exp_ratios_df):
    """
    :param results_df: The input data frame.  It should contains columns for num_chains, chain_length, starting_state,
    and columns for all occurrences and end occurrences for each state
    :param num_chains:
    :param chain_length:
    :param exp_ratios_df: a df containing the "expected" ratios of each state based on the calculated probability; 
    columns are "state" and "probability"
    :return:
    """
    df_matching_params = results_df[(results_df.num_chains == num_chains) & (results_df.chain_length == chain_length)]
    #abc = df_matching_params[['starting_state', 'A_all', 'B_all', 'C_all', 'A_end', 'B_end', 'C_end']].groupby(
    #    'starting_state').sum()
    ac = pd.Series(df_matching_params.columns)
    #ac = ac[ac.apply(lambda c: "all" in c)]
    #ac = ac[ac.apply(lambda c: "_end" not in c)]
    ac = ac[ac.apply(lambda c: c.endswith("_all"))]
    ec = pd.Series(results_df.columns)
    ec = ec[ec.apply(lambda c: c.endswith("_end"))]
    abc = df_matching_params[['starting_state'] + list(ac) + list(ec)].groupby(
        'starting_state').sum()

    # Check that the "all" columns and "end" columns match
    assert set(ec.str.replace("_end", "")) == set((ac.str.replace("_all", "")))
    # TODO:  Should we check the order also?

    exp_ratios = exp_ratios_df.probability
    exp_numbers = [num_chains * chain_length * df_matching_params.shape[0] * r for r in exp_ratios] + \
                  [num_chains * df_matching_params.shape[0] * r for r in exp_ratios]

    exp_ratios_df['all_counts'] = exp_ratios_df.apply(lambda row: row.probability * num_chains * chain_length *
                                                  df_matching_params.shape[0], axis=1)
    exp_ratios_df['end_counts'] = exp_ratios_df.apply(lambda row: row.probability * num_chains *
                                                               df_matching_params.shape[0], axis=1)

    # sort them by the state values so that they match up
    exp_ratios_df = exp_ratios_df.sort_values('state')
    abc = abc.sort_index()

    states_list = list(ec.str.replace("_end", ""))
    result = {}
    for starting_state in states_list:
        '''result[starting_state + " all, exp"] = chisquare(abc.loc[starting_state][ac], np.array(exp_ratios_df.all_counts /
                                                                             abc.shape[0])).pvalue
        result[starting_state + " end, exp"] = chisquare(abc.loc[starting_state][ec], np.array(exp_ratios_df.end_counts /
                                                                             abc.shape[0])).pvalue'''
        # not needed for the current hypotheses being tested
        pass


    pairings = get_pairings(states_list)
    for pair in pairings:
        '''all_name = pair[0] + " all, " + pair[1] + " all"
        result[all_name] = chisquare(abc.loc[pair[0]][ac], abc.loc[pair[1]][ac]).pvalue
        end_name = pair[0] + " end, " + pair[1] + " end"
        result[end_name] = chisquare(abc.loc[pair[0]][ec], abc.loc[pair[1]][ec]).pvalue'''
        # This chisquares results of each starting room to all the other starting rooms.  It is interesting but it
        # not one of the hypotheses being examined, so commenting it out.
        pass

    #result['all'] = chisquare(abc.sum(), exp_ratios_df)
    total_all = abc[ac].sum()#.sort_values(ascending=False)
    total_end = abc[ec].sum()#.sort_values(ascending=False)
    #total_all.apply(lambda a: result[a] = a)

    result['hyp 1 all'] = chisquare(total_all, exp_ratios_df.all_counts).pvalue
    result['hyp 2 end'] = chisquare(total_end, exp_ratios_df.end_counts).pvalue
    result['hyp 3'] = chisquare(total_all.values, (total_end * chain_length).values).pvalue
    for column in abc.columns:
        result['hyp 4 ' + column] = chisquare(abc[column]).pvalue
    return result


def get_pairings(states_list):
    pairings = []
    """for item in states_list:
        for level_2_item in states_list:
            if item != level_2_item:
                pairings.append((item, level_2_item))"""
    for i in range(len(states_list)):
        for j in range(i + 1, len(states_list)):
            pairings.append((states_list[i], states_list[j]))
    return pairings


def run_house_sim():
    dist_map = get_nine_rooms_dist_map()
    num_cycles = 10
    for i in range(num_cycles):
        once_from_each_starting_state(100, 10000, dist_map, save_results_9_states)


def save_results_9_states(all_states_hist, ending_states_hist, starting_state):
    save_results(all_states_hist, ending_states_hist, starting_state, "sim_results/9_states_results.csv")


def show_3_states_results():
    df = pd.read_csv("sim_results/3_states_results.csv")
    exp_ratios = [.3846, .3846, .2308]
    exp_ratios_df = pd.DataFrame({"state": ['A', 'B', 'C'], "probability": exp_ratios})
    # print(get_stats_for_params_3_state(df, 1000, 1000, exp_ratios))
    # print(get_stats_for_params_3_state(df, 10, 100000, exp_ratios))
    stats_1000_1000 = get_stats_for_params(df, 1000, 1000, exp_ratios_df)
    stats_10_100000 = get_stats_for_params(df, 10, 100000, exp_ratios_df)

    print("1000, 1000")
    for key, value in stats_1000_1000.items():
        print(key, value)

    print("\n10, 100000")
    for key, value in stats_10_100000.items():
        print(key, value)


def show_9_states_results():
    df = pd.read_csv("sim_results/9_states_results.csv")
    df = df.drop(23, axis=0)
    exp_ratios_df = rooms_probs.compute_probs()
    # print(get_stats_for_params_3_state(df, 1000, 1000, exp_ratios))
    # print(get_stats_for_params_3_state(df, 10, 100000, exp_ratios))
    exp_ratios_df = exp_ratios_df.rename(columns={"room": "state"})
    stats_100_10000 = get_stats_for_params(df, 100, 10000, exp_ratios_df)

    print("1000, 1000")
    for key, value in stats_100_10000.items():
        print(key, value)


if __name__ == "__main__":
    #test_run()


    #run_house_sim()
    print("\n3")
    #show_3_states_results()
    print("\n9")
    show_9_states_results()
