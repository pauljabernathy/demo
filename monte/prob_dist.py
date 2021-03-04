import numpy as np
import xarray as xr

class ProbDist:
    #TODO: a hash or UUID to make a unique identifier; a function to create a default if none is given
    def __init__(self, data:dict = None, values = None, probs: list = None, id=None, confidence=None):
        """if a data dictionary is given, those values are used
        otherwise, it uses items and probs

        """
        self.id = id
        if data is not None:
            self.data = data
            self._set_variables(self.data)
        elif values is not None and probs is not None:
            kv_pair_zip = zip(values, probs)
            self.data = {}
            for pair in kv_pair_zip:
                if pair[0] not in self.data:
                    self.data[pair[0]] = pair[1]
                else:
                    self.data[pair[0]] += pair[1]
            self._set_variables(self.data)

            self.dimensions = [self.id]

            """"#TODO: make values to be a set, and unit test the case when there are repeated values
            self.items = set(items)
            #TODO: If there were non unique values, combine the probs.
            self.probs = self.normalize_list(np.array(probs)) if auto_normalize else np.array(probs)
            if len(items) != len(probs):
                raise Exception('The set of unique values and probabilities must be equal length.')
            self.data = {k: v for k, v in zip(items, probs)}"""
        else:
            raise Exception("You must provide items values and probabilities.")

    def __str__(self):
        return self.id

    def _set_variables(self, data:dict):
        self.confidence = sum(data.values())
        self.data = ProbDist.normalize_data(self.data)
        self.values = np.array(list(data.keys()))
        self.probs = np.array(list(data.values()))
        # Jan 2021 - starting to experiment with adding data in the form of a numpy array.  If it works,
        # I might replace the dictionary entirely with the np array.
        self.np_data = ProbDist.get_data_array(self.data)
        # Feb 2021 - moving to xarray to be able to handle multi dimensional data without having to keep track of the
        # axis labels and value labels myself.
        self.xr_data = xr.DataArray(list(self.data.values()), dims=(self.id), coords={self.id: list(self.data.keys())})
        x = 5

    @staticmethod
    def get_data_array(data_dict):
        values = list(data_dict.values())
        array = np.zeros(len(values))
        for i in range(len(values)):
            array[i] = values[i]
        return array

    def verify_probabilities(self):
        return self.probs.sum() == 1.0

    @staticmethod
    def normalize_data(data:dict):
        total = sum(data.values())
        for kv_pair in data.items():
            data[kv_pair[0]] = kv_pair[1] / total
        #self.probs = self.probs / total
        return data

    @staticmethod
    def normalize_list(probs):
        """list should be a list or numpy array of numbers
        returns a numpy array
        """
        if isinstance(probs, list):
            #This way should work with an nd array but the other way should be faster, in theory.
            total = sum(probs)
            return [x / total for x in probs]
        elif isinstance(probs, np.ndarray):
            return probs / probs.sum()

    # TODO:  What is this for?  What does "confidence" mean?  Why would you "denormalize" inplace?  I thought you
    #  should never have probability values that do not add up to 1.
    @staticmethod
    def denormalize(probs_dict, confidence, inplace=False):
        if inplace:
            denormalized_data = probs_dict
        else:
            denormalized_data = {}
        for k, v in probs_dict.items():
            denormalized_data[k] = probs_dict[k] * confidence
        return denormalized_data

    def get(self, key):
        if key not in self.data:
            return 0
        else:
            return self.data[key]

    def get_random_value(self, num_values=1):
        return np.random.choice(a=self.values, size=num_values, p=self.probs)

    #TODO: somethin more like a query or match criteria, not a list of possible values
    def get_random_value_conditional(self, values, num_values, include=True):
        """gets a random value, based on the include or exclude list given
        values: the values to choose from, or exclude; all should be in the distribution
        include: If this is true, only select from the values given.  If false, select from the values not in this list.
        """
        if include:
            return self._get_random_values_include_list(values, num_values)
        else:
            return self._get_random_values_exclude_list(values, num_values)

    def _get_random_values_include_list(self, values, num_values):
        #a couple of methods 1) some sort of boolean mask to match up things in values and self.values, and apply it to self.probs, then normalize
        #2) iterate through self.data and put into a new dict; 1 seems faster but 2 is more straightforward
        #values_to_include = self.values.intersection(set(values))

        cond_data = {}
        for k, v in self.data.items():
            if k in values:
                cond_data[k] = v
        cond_data = ProbDist.normalize_data(cond_data)
        return np.random.choice(a=list(cond_data.keys()), p=list(cond_data.values()), size=num_values)

    def _get_random_values_exclude_list(self, values, num_values):
        cond_data = {}
        for k, v in self.data.items():
            if k not in values:
                cond_data[k] = v
        cond_data = ProbDist.normalize_data(cond_data)
        return np.random.choice(a=list(cond_data.keys()), p=list(cond_data.values()), size=num_values)

    def _get_random_values_include_condition(self, condition, num_values=1):
        indices = np.apply_along_axis(condition, 0, self.values)
        return np.random.choice(a=self.values[indices], p = ProbDist.normalize_list(self.probs[indices]), size=num_values)

    def merge(self, additional_data_dict:dict, inplace=True):
        """
        Merge additional data with this ProbDist.  Where keys are equal, the probabilities will be combined.  Where
        the new data has new keys, they will be added.
        :param additional_probs_dict: denormalized data to merge with this ProbDist
        :return:
        """
        # TODO: ability merge with another ProbDist
        new_data = self.denormalize(self.data, self.confidence, inplace)
        for k, v in additional_data_dict.items():
            new_data[k] += additional_data_dict[k]

        new_data = self.normalize_data(new_data)
        if inplace:
            self.data = new_data
            self._set_variables(self.data)
        return ProbDist(new_data)

    def join(self, other):
        """
        find joint distribution with another ProbDist
        :param other:
        :return:
        """
        new_data = {}
        for self_key, self_value in self.data.items():
            for other_key, other_value in other.data.items():
                # TODO: This should be a set or a string where the values are sorted, or something so the order is
                #  the same whichever of the two ProbDists is first.
                # dist1.join(dist2) should be the same as dist2.join(dist1)
                #new_data[str(self_key) + "," + str(other_key)] = self_value * other_value
                new_data[self.get_key(self_key, other_key)] = self_value * other_value
                # If both are legitimate ProbDists, there there should be no replicated self_key,other_key combintations,
                # so _should_ be no need to check.

        id = ProbDist.get_key(self.id, other.id)
        result = ProbDist(new_data, id=id)

        # now the np array
        # TODO:  Decide if this is the best way to handle the dimensions.
        '''
        Some options:
            - this way, alphabetically
            - just use the join order
            - have the user specify
            - one of the above as the default, but allow the user to specify
        Ideally, it won't matter a whole lot if at all because I wouldn't expect the user to get the np_data out 
        directly.  So long as ProbDist can keep it straight and provide P(X = x | Y = y) or ProbDist(given Z = z), 
        it shouldn't matter.
        '''
        dims = (len(self.data.values()), len(other.data.values()))
        dimensions = id.split(',')
        if dimensions[0] == self.id:
            left = self
            right = other
        else:
            left = other
            right = self
        np_data = ProbDist.cartesian_product(left.probs, right.probs)
        result.np_data = np_data
        result.xr_data = xr.DataArray(np_data.reshape(dims), dims=(self.id, other.id),
                                      coords={self.id: list(self.data.keys()), other.id: list(other.data.keys())})
        return result

    @staticmethod
    def get_key(key1: str, key2: str):
        key1 = str(key1)
        key2 = str(key2)
        values_list = [key1, key2]
        values_list.sort()
        we_want_quotes = False
        if we_want_quotes:
            return str(values_list).replace('[', '').replace(']', '').replace(" ", '')
        else:
            if key2 < key1:
                return key2 + "," + key1
            else:
                return key1 + "," + key2

    # TODO: Make sure this works for more than two dimensions
    @staticmethod
    def cartesian_product(x, y):
        return np.array([x0 * y0 for x0 in x for y0 in y]).reshape(len(x), len(y))
        #result = np.zeros()


def __eq__(self, other):
        if type(other) is not ProbDist:
            return False
        #if other.data.size != self.data.size:
        #    return False
        return other.data == self.data


class ConditionalProbDist:
    pass


# TODO:  multidimensional data
"""
Plan:
leaning toward using xarray
When joining ProbDists, the cartesian product needs to be able to handle more than one dimension.
So multiple every cell in the left by every cell in the right, put into an np array, then reshape to the correct 
dimensions.
Use the ProbDist ids for the dimension names of the xarray, and the values for the "coords" :parameter.
Xarray, like numpy, has outer dimensions and inner dimensions, and that determines how you will access data in the 
array.  To the user of the ProbDist, I do not want that to be an issue.  I guess you can get the data out, 
but really I want most of the accessing of the cells to be specifying each dimension and value, order not mattering.

You should be able to specify the numbers directly and not solely rely on a join, because in the join they 
probabilities are still independent so P(X = x | Y = y) just gives you back the original P(X = x).  It is when the 
dimensions interact, when they are not independent, that interesting things can happen.

"""