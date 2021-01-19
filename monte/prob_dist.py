import numpy as np

class ProbDist:
    #TODO: a hash or UUID to make a unique identifier; a function to create a default if none is given
    def __init__(self, data:dict = None, values = None, probs: list = None, id=None, confidence=None):
        """if a data dictionary is given, those values are used
        otherwise, it uses items and probs

        """
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


            """"#TODO: make values to be a set, and unit test the case when there are repeated values
            self.items = set(items)
            #TODO: If there were non unique values, combine the probs.
            self.probs = self.normalize_list(np.array(probs)) if auto_normalize else np.array(probs)
            if len(items) != len(probs):
                raise Exception('The set of unique values and probabilities must be equal length.')
            self.data = {k: v for k, v in zip(items, probs)}"""
        else:
            raise Exception("You must provide items values and probabilities.")
        self.id = id

    def __str__(self):
        return self.id

    def _set_variables(self, data:dict):
        self.confidence = sum(data.values())
        self.data = ProbDist.normalize_data(self.data)
        self.values = np.array(list(data.keys()))
        self.probs = np.array(list(data.values()))


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

    @staticmethod
    def denormalize(probs_dict, confidence, inplace=False):
        if inplace:
            denormalized_data = probs_dict
        else:
            denormalized_data = {}
        for k, v in probs_dict.items():
            denormalized_data[k] = probs_dict[k] * confidence
        return denormalized_data


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
        new_data = self.denormalize(self.data, self.confidence, inplace)
        for k, v in additional_data_dict.items():
            new_data[k] += additional_data_dict[k]

        new_data = self.normalize_data(new_data)
        if inplace:
            self.data = new_data
            self._set_variables(self.data)
        return ProbDist(new_data)


class ConditionalProbDist:
    pass