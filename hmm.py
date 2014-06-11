# Hidden Markov Model for Part of Speech Tagging

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

if __name__ == "__main__":
    order = 1
    tagger = POSTagger()
    infile = 'assignment3/allTraining.txt'
    with open(infile, 'r') as f:
        tagger.train(f.read().split(), order)

