# Language Learning Model


import random

class ContextCounter(object):
    def __init__(self):
        self.token_counter = {}
        self.count = 0

class LanguageLearner(object):
    def __init__(self):
        self.model = {}
        self.total = 0
    
    def get_const_context(self):
        return ['']
    
    def train(self, tokens, order):
        print "training"
        context = self.get_const_context()
        for token in tokens:
            token = token.split('_')[0]
            token = ''.join(c for c in token if c.isalpha())
            if token == '':
                continue
            self.model[str(context)] = self.model.setdefault(str(context), ContextCounter())
            self.model[str(context)].token_counter[token] = self.model[str(context)].token_counter.setdefault(token, 0) + 1
            self.model[str(context)].count += 1
            self.total += 1
            if self.total % 100000 == 0:
                print self.total
            if len(context) < order:
                context = context + [token]
            else:
                context = (context + [token])[1:]
    
    def generate_text(self, number_of_tokens, order):
        print "generating"
        context = self.get_const_context()
        for i in xrange(number_of_tokens):
            word_index = random.randint(1, self.model[str(context)].count)
            prev_total = 0
            curr_total = 0
            word = ''
            for key, value in self.model[str(context)].token_counter.iteritems():
                curr_total += value
                if prev_total < word_index and curr_total >= word_index:
                    word = key
                    break
                prev_total = curr_total
            print(word),
            if len(context) < order:
                context = context + [word]
            else:
                context = (context + [word])[1:]

if __name__ == '__main__':
    order = 2
    ll = LanguageLearner()
    
    infile = 'assignment3/allTraining.txt'
    with open(infile, 'r') as f:
        ll.train(f.read().split(), order)
    
    ll.generate_text(100, order)
        
    
    
