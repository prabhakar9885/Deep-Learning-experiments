import os, re, sys
import Aux
from gensim.models import Word2Vec

def getsimiliar(name):
    res = []
    for word in model.most_similar(Aux.StemmingHelper.stem(name)):
        res.append( "{}: {}".format( Aux.StemmingHelper.original_form(word[0].strip()), word[1] ) )
    return res;



dirpath = sys.argv[1]
sentences = []

print( "Reading files")
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

print( "No. of lines: {}".format(len(sentences)))

min_count = 2
size = 50
window = 4
print( "Building word2vec model");
model = Word2Vec(sentences, min_count=min_count, size=size, window=window)


print("Generating similar words for characters.txt");
fin = open("./characters.txt")
fout = open("./word2vec.out", "w")
for line in fin:
    name = line.split()[0]
    name = Aux.StemmingHelper.stem(name.lower())
    fout.write( "{}\n".format(name) )
    fout.writeline( "-"*len(name) )
    try:
        fout.write( "\n".join(getsimiliar(name)) )
        fout.write( "\n" )
    except:
        pass


fout.close()
print("Done")
