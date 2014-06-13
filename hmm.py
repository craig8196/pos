# Hidden Markov Model for Part of Speech Tagging
import math

UNKNOWN = 'UNKNOWN_IDENTIFIER'

class Counter(object):
    def __init__(self):
        self.pos_count = {}
        self.pos_count[UNKNOWN] = 1
        self.total_pos_count = 1
        
        self.evidence_count = {}
        self.evidence_count[UNKNOWN] = 1
        self.total_evidence_count = 1
    
    def add_transition(self, pos):
        self.pos_count[pos] = self.pos_count.setdefault(pos, 0) + 1
        self.total_pos_count += 1
    
    def add_evidence(self, evidence):
        self.evidence_count[evidence] = self.evidence_count.setdefault(evidence, 0) + 1
        self.total_evidence_count += 1
    
    def get_total_pos_count(self):
        return self.total_pos_count
    
    def clear_all_counts(self):
        self.pos_count = {}
        self.total_pos_count = 0
        
        self.evidence_count = {}
        self.total_evidence_count = 0
    
    def pos_count_counts(self, counter):
        for pos in counter.pos_count:
            count = counter.pos_count[pos]
            self.pos_count[pos] = self.pos_count.setdefault(pos, 0) + count
            self.total_pos_count += count
    
    def compute_pos_log_prob(self):
        self.pos_log = {}
        self.total_pos_log = math.log(self.total_pos_count)
        for pos, count in self.pos_count.items():
            self.pos_log[pos] = math.log(count) - self.total_pos_log
    
    def compute_evidence_log_prob(self):
        self.evidence_log = {}
        self.total_evidence_log = math.log(self.total_evidence_count)
        for ev, count in self.evidence_count.items():
            self.evidence_log[ev] = math.log(count) - self.total_evidence_log
    
    def get_full_pos_log_prob(self, pos):
        if pos not in self.pos_count:
            pos = UNKNOWN
        return self.pos_log[pos]
    
    def get_full_evidence_log_prob(self, evidence):
        if evidence not in self.evidence_log:
            evidence = UNKNOWN
        return self.evidence_log[evidence]

class Path(list):
    def __init__(self):
        super(list, self).__init__()
        self.probability = 0

class POSTagger(object):
    def __init__(self):
        self.model = {} # pos model
        self.model[UNKNOWN] = Counter()
        self.total_pos = Counter()
        self.total_pos.clear_all_counts()

    def get_const_context(self, order=1):
        if order == 1:
            return '.'
        else:
            result = []
            for i in range(order):
                result.append('')
            return result

    def train(self, tokens, order=1):
        print "Training..."
        context = '.'
        for token in tokens:
            evidence_token, pos_token = token.split('_')
            self.model[context] = self.model.setdefault(context, Counter())
            mapper = self.model[context]
            mapper.add_transition(pos_token)
            mapper.add_evidence(evidence_token)
            context = pos_token
        
        # find total pos token counts for computing P(Z=pos_token)
        # compute log probabilities
        print "Computing totals and logs..."
        for k, v in self.model.items():
            self.total_pos.pos_count_counts(v)
            v.compute_pos_log_prob()
            v.compute_evidence_log_prob()
        self.total_pos.compute_pos_log_prob()
        
        print self.total_pos.pos_count
        print self.total_pos.total_pos_count

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
    
    def get_log_evidence_given_pos(self, evidence, pos):
        if pos not in self.model:
            pos = UNKNOWN
        return self.model[pos].get_full_evidence_log_prob(evidence)
        
    
    def test(self, tokens, order=1):
        if len(tokens)<1:
            raise Exception("Can't test zero tokens.")
        
        print "Testing..."
        
        # initialize starting states
        evidence_token, pos_token = tokens[0].split('_')
        paths = {}
        correct_path = Path() + [pos_token]
        for pos in self.total_pos.pos_count:
            p = Path()
            p.append(pos)
            paths[pos] = p
            paths[pos].probability = self.total_pos.get_full_pos_log_prob(pos) + \
                                     self.get_log_evidence_given_pos(evidence_token, pos_token)
        
        print paths
        raise Exception('stop')
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
                + self.handle_unknown(prob_map.pos_count, h_state) - 2 * math.log(prob_map.total_count), h_state) for h_state in self.model.keys())
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

    outfile = 'assignment3/devtest.txt'
    with open(outfile, 'r') as f:
        tagger.test(f.read().split(), order)

