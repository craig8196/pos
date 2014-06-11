# Hidden Markov Model for Part of Speech Tagging
import math


class ProbabilityMapper(object):
    def __init__(self):
        self.transition_count = {}
        self.sensor_count = {}
        self.total_count = 0

class POSTagger(object):
    def __init__(self):
        self.model = {}
        self.total = 0

    def get_const_context(self):
        return ['']

    def train(self, tokens, order):
        print "training"
        context = self.get_const_context()
        for token in tokens:
            temp_token_split = token.split('_')
            evidence_token = temp_token_split[0]
            pos_token = temp_token_split[1]
            self.model[str(context)] = self.model.setdefault(str(context), ProbabilityMapper())
            self.model[str(context)].transition_count[pos_token] = self.model[str(context)].transition_count.setdefault(pos_token, 0) + 1
            self.model[str(context)].sensor_count[evidence_token] = self.model[str(context)].sensor_count.setdefault(evidence_token, 0) + 1
            self.model[str(context)].total_count += 1
            self.total += 1
            if self.total % 100000 == 0:
                print self.total
            if len(context) < order:
                context = context + [pos_token]
            else:
                context = (context + [pos_token])[1:]

    def handle_unknown(self, count_map, token):
        if token in count_map:
            return math.log(count_map[token])
        else:
            return 0

    def compare_tags(self, tokens, chosen_sequence):
        total_correct = 0
        total = len(tokens)
        for i in range(0, len(tokens)):
            pos_token = tokens[i].split('_')[1]
            pos_token = "['" + pos_token + "']"
            if i == 1:
                print pos_token
                print chosen_sequence[i]
            if pos_token == chosen_sequence[i]:
                total_correct += 1
        print total_correct * 1.0 / total 

    def test(self, tokens):
        print "testing"
        self.state_value_holder = [{}]
        self.sequence_keeper = {}

        num_of_observations = len(tokens)
        num_of_training_states = len(self.model)
        first_token = tokens[0].split('_')[0]
        for hidden_state, prob_map in self.model.iteritems():
            #start probability for the state is the occurences for that state divided by the total number of state occurrences (from training data)

            #the numerator of the start probability and the denominator of the sensor model 
            #cancel each other out (the number of occurences of the state in the training data) 
            self.state_value_holder[0][hidden_state] = self.handle_unknown(prob_map.sensor_count, first_token) - math.log(self.total)
            self.sequence_keeper[hidden_state] = [hidden_state]

        for obs_i in range(1, num_of_observations):
            if obs_i % 1000 == 0:
                print obs_i
            self.state_value_holder.append({})
            new_sequence = {}
            evi_token = tokens[obs_i].split('_')[0]

            for hidden_state, prob_map in self.model.iteritems():
                (state_value, state) = max((self.state_value_holder[obs_i - 1][h_state] + self.handle_unknown(prob_map.sensor_count, evi_token) \
                + self.handle_unknown(prob_map.transition_count, h_state) - 2 * math.log(prob_map.total_count), h_state) for h_state in self.model.keys())
                self.state_value_holder[obs_i][hidden_state] = state_value
                new_sequence[hidden_state] = self.sequence_keeper[state] + [hidden_state]

            self.sequence_keeper = new_sequence

        (state_value, state) = max((self.state_value_holder[obs_i][h_state], h_state) for h_state in self.model.keys())
        self.best_sequence = self.sequence_keeper[state]
        self.compare_tags(tokens, self.best_sequence)


if __name__ == "__main__":
    order = 1
    tagger = POSTagger()

    infile = 'assignment3/allTraining.txt'
    with open(infile, 'r') as f:
        tagger.train(f.read().split(), order)
    # print tagger.model["['DT']"].transition_count["NN"]

    outfile = 'assignment3/devtest.txt'
    with open(outfile, 'r') as f:
        tagger.test(f.read().split())

