scriptdir='stanford-corenlp-full-2018-02-27'
# en discourse path
PARSE_PATH='data/raw_txt'

# train
for F_NAME in $PARSE_PATH/*.out
do
    XML_F_PATH="$F_NAME".xml
    XML_F_NAME=${XML_F_PATH##*/}
    if [ ! -f $XML_F_PATH ]; then
        cpulimit -l 500 /usr/bin/java -mx2g -cp "$scriptdir/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -ssplit.eolonly -tokenize.whitespace true -file $F_NAME
        mv $XML_F_NAME $XML_F_PATH
    fi
done
