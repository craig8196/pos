# Language Learning Model


import random

class LanguageLearner(object):
    def __init__(self):
        self.model = {}
    
    def get_const_context(self):
        return ['']
    
    def train(self, tokens):
        context = self.get_const_context()
        for token in tokens:
            model[str(context)] = model.setdefault(str(context), []) + [token]
            context = (context + [token])[1:]
    
    def generate_text(self, number_of_tokens):
        context = self.get_const_context()
        for i in xrange(number_of_tokens):
            word = random.choice(model[str(context)])
            print(word, end=' ')
            context = (context+[word])[1:]
        print()



if __name__ == '__main__':
    ll = LanguageLearner()
    
    infile = ''
    with open(infile, 'r') as f:
        ll.train(f.read().split())
    
    ll.generate_text(100)
        
    
    
