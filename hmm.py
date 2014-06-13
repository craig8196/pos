# Hidden Markov Model for Part of Speech Tagging
from __future__ import division
import sys
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
        if pos not in self.pos_count:
            self.pos_count[pos] = 1
            self.total_pos_count += 1
        self.pos_count[pos] += 1
        self.total_pos_count += 1
    
    def add_evidence(self, evidence):
        if evidence not in self.evidence_count:
            self.evidence_count[evidence] = 1
            self.total_evidence_count += 1
        self.evidence_count[evidence] += 1
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
        
    def evidence_count_counts(self, counter):
        for ev, count in counter.evidence_count.items():
            self.evidence_count[ev] = self.evidence_count.setdefault(ev, 0) + count
            self.total_evidence_count += count
    
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
        if pos not in self.pos_log:
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

PUNCTUATION = {
    '`': '``',
    '``': '``',
    "'": "''",
    "''": "''",
    '(': '(',
    '[': '(',
    '{': '(',
    ')': ')',
    ']': ')',
    '}': ')',
    ',': ',',
    '--': '--',
    '.': '.',
    '!': '.',
    '?': '.',
    ':': ':',
    ';': ':',
    '...': ':',
}

class POSTagger(object):
    def __init__(self):
        self.model = {} # pos model
        self.double_model = {}
        self.double_emit = {}
        self.totals = Counter()
        self.totals.clear_all_counts()
        self.double_totals = Counter()
        self.double_totals.clear_all_counts()

    def train(self, tokens):
        print "Training..."
        context = '.'
        double_context = (tokens[-2].split('_')[1], tokens[-1].split('_')[1])
        for token in tokens:
            evidence_token, pos_token = token.split('_')
            evidence_token = evidence_token.lower()
            if context not in self.model:
                self.model[context] = Counter()
            if pos_token not in self.model:
                self.model[pos_token] = Counter()
            if double_context not in self.double_model:
                self.double_model[double_context] = Counter()
            self.model[context].add_transition(pos_token)
            self.model[pos_token].add_evidence(evidence_token)
            self.double_model[double_context].add_transition(pos_token)
            context = pos_token
            double_context = (double_context[1], pos_token)
        
        # compute log probabilities and compile totals
        print "Computing totals and logs..."
        for k, v in self.model.items():
            self.totals.pos_count_counts(v)
            self.totals.evidence_count_counts(v)
            v.compute_pos_log_prob()
            v.compute_evidence_log_prob()
        self.totals.compute_pos_log_prob()
        self.totals.compute_evidence_log_prob()
        self.model[UNKNOWN] = self.totals
        
        for k, v in self.double_model.items():
            self.double_totals.pos_count_counts(v)
            v.compute_pos_log_prob()
        self.double_totals.compute_pos_log_prob()
        self.double_model[UNKNOWN] = self.double_totals
        
        #~ print self.totals.pos_count
        #~ print self.totals.total_pos_count
    
    def get_log_evidence_given_pos(self, evidence, pos):
        if pos not in self.model:
            pos = UNKNOWN
        return self.model[pos].get_full_evidence_log_prob(evidence)
    
    def get_context_counts(self, context):
        if context not in self.model:
            context = UNKNOWN
        return self.model[context]
    
    def get_double_context_counts(self, double_context):
        if double_context not in self.double_model:
            double_context = UNKNOWN
        return self.double_model[double_context]
    
    def get_max_path(self, paths):
        index = UNKNOWN
        for key in paths:
            if paths[key].probability > paths[index].probability:
                index = key
        return paths[index]
    
    def get_log(self, context, next_pos, evidence_token):
        counts = self.get_context_counts(context)
        next_counts = self.get_context_counts(next_pos)
        return counts.get_full_pos_log_prob(next_pos)+next_counts.get_full_evidence_log_prob(evidence_token)
    
    def get_double_log(self, double_context, next_pos, evidence_token):
        counts = self.get_double_context_counts(double_context)
        next_counts = self.get_context_counts(next_pos)
        return counts.get_full_pos_log_prob(next_pos)+next_counts.get_full_evidence_log_prob(evidence_token)
    
    def initialize(self, token, paths):
        evidence_token, pos_token = token.split('_')
        evidence_token = evidence_token.lower()
        double_paths = {}
        correct_path = Path() + [pos_token]
        for pos in self.totals.pos_count:
            p = Path()
            p.append(pos)
            paths[pos] = p
            paths[pos].probability = self.totals.get_full_pos_log_prob(pos) + \
                                     self.get_log_evidence_given_pos(evidence_token, pos_token)
        return correct_path
    
    def first_order_iterate(self, token, paths):
        evidence_token, pos_token = token.split('_')
        evidence_token = evidence_token.lower()
        
        for key, path in paths.items():
            context = path[-1]
            prob = path.probability
            max_prob = -sys.float_info.max
            max_token = UNKNOWN
            counts = self.get_context_counts(context)
            # find path with max probability
            for next_pos in counts.pos_count:
                new_prob = prob + self.get_log(context, next_pos, evidence_token)
                if new_prob > max_prob:
                    max_prob = new_prob
                    max_token = next_pos
            path.probabiliy = max_prob
            path.append(max_token)
    
    def second_order_iterate(self, token, paths):
        evidence_token, pos_token = token.split('_')
        evidence_token = evidence_token.lower()
        
        for key, path in paths.items():
            double_context = (path[-2], path[-1])
            prob = path.probability
            max_prob = -sys.float_info.max
            max_token = UNKNOWN
            counts = self.get_double_context_counts(double_context)
            # find path with max probability
            for next_pos in counts.pos_count:
                new_prob = prob + self.get_double_log(double_context, next_pos, evidence_token)
                if new_prob > max_prob:
                    max_prob = new_prob
                    max_token = next_pos
            path.probabiliy = max_prob
            path.append(max_token)
    
    def confusion_matrix(self, name, mypath, correctpath, tokens):
        length = len(mypath)
        if len(correctpath) < length:
            length = len(correctpath)
        print name
        confused = {}
        for i in range(length):
            if mypath[i] != correctpath[i]:
                key = (mypath[i], correctpath[i])
                confused[key] = confused.setdefault(key, 0) + 1
        for i in range(20):
            m = max((v, conf) for conf, v in confused.items())
            print m[1], m[0]
            del(confused[m[1]])
            if len(confused) == 0:
                break
    
    def test(self, tokens):
        if len(tokens)<2:
            raise Exception("Can't test one or fewer tokens.")
        
        print "Testing..."
        
        print "Token count:", len(tokens)
        
        # initialize starting states
        evidence_token, pos_token = tokens[0].split('_')
        evidence_token = evidence_token.lower()
        paths = {}
        double_paths = {}
        correct_path = self.initialize(tokens[0], paths)
        self.initialize(tokens[0], double_paths)
        
        self.first_order_iterate(tokens[1], double_paths)
        
        total_first_order_iterations = 1000
        total_second_order_iterations = 1000
        
        for i in range(1, total_first_order_iterations):
            self.first_order_iterate(tokens[i], paths)
            correct_path.append(tokens[i].split('_')[1])
        
        for i in range(2, total_second_order_iterations):
            self.second_order_iterate(tokens[i], double_paths)
        
        correct = 0
        max_path = self.get_max_path(paths)
        for i in range(len(max_path)):
            if correct_path[i] == max_path[i]:
                correct += 1
        print "First order correct:", correct/total_first_order_iterations
        self.confusion_matrix('first', max_path, correct_path, tokens)
        
        correct = 0
        max_path = self.get_max_path(double_paths)
        for i in range(len(max_path)):
            if correct_path[i] == max_path[i]:
                correct += 1
        print "Second order correct:", correct/total_second_order_iterations
        self.confusion_matrix('second', max_path, correct_path, tokens)

if __name__ == "__main__":
    tagger = POSTagger()

    infile = 'assignment3/allTraining.txt'
    with open(infile, 'r') as f:
        tagger.train(f.read().split())

    outfile = 'assignment3/devtest.txt'
    with open(outfile, 'r') as f:
        tagger.test(f.read().split())

