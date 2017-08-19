import os, re, sys
from glove import Corpus, Glove
import Aux


def getsimiliar(name):
    res = []
    for word in glove.most_similar(Aux.StemmingHelper.stem(name,9)):
        res.append( "{}: {}".format( Aux.StemmingHelper.original_form(word[0].strip()), word[1] ) )
    return res;

dirpath = sys.argv[1]
sentences = []

for filename in os.listdir(dirpath):
    filename = "{}{}".format(dirpath,filename)
    f = open( filename )
    for l in f:
        words = re.split( r"\W", l )
        stems = []
        for word in words:
            stem = Aux.StemmingHelper.stem(word.lower())
            if len(stem) > 0:
                stems.append(stem)
        if len(stems) > 0:
            sentences.append( stems )


corpus = Corpus()
corpus.fit(sentences, window=8)

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=50, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

fin = open("./characters.txt")
fout = open("./word2vec.out", "w")
print( "No. of lines: {}\n".format(len(sentences)))
for line in fin:
    name = line.split()[0]
    name = Aux.StemmingHelper.stem(name.lower())
    fout.write( "{}\n".format(name) )
    fout.writelines( "-"*len(name) +"\n" )
    try:
        fout.write( "\n".join(getsimiliar(name)) )
        fout.write( "\n" )
    except:
        pass


fout.close()
