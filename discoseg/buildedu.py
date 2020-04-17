from os import listdir
from os.path import join, basename
from discoseg.model.classifier import Classifier
from discoseg.model.docreader import DocReader
from discoseg.model.sample import SampleGenerator
import gzip
import _pickle as cPickle
import multiprocessing as mp
from utils.xmlreader import reader, writer, combine
from os import listdir
from os.path import join
from config import *

rpath_ = ""
wpath_ = ""
clf_ = Classifier()
vocab_ = None
dr_ = None


def extract(rpath):
    files = [join(rpath, fname) for fname in listdir(rpath) if fname.endswith(".xml")]
    for fxml in files:
        print('Processing file: {}'.format(fxml))
        sentlist, constlist = reader(fxml)
        sentlist = combine(sentlist, constlist)
        fconll = fxml.replace(".xml", ".conll")
        writer(sentlist, fconll)


def do_seg_one_(fname):
    doc = dr_.read(fname, withboundary=False)
    sg = SampleGenerator(vocab_)
    sg.build(doc)
    M, _ = sg.getmat()
    predlabels = clf_.predict(M)
    doc = postprocess(doc, predlabels)
    writedoc(doc, fname, wpath_)


def postprocess(doc, predlabels):
    """ Assign predlabels into doc
    """
    tokendict = doc.tokendict
    for gidx in tokendict.keys():
        if predlabels[gidx] == 1:
            tokendict[gidx].boundary = True
        else:
            tokendict[gidx].boundary = False
        if tokendict[gidx].send:
            tokendict[gidx].boundary = True
    return doc


def writedoc(doc, fname, wpath):
    """ Write file
    """
    tokendict = doc.tokendict
    N = len(tokendict)
    fname = basename(fname).replace(".conll", ".merge")
    fname = join(wpath, fname)
    eduidx = 1
    with open(fname, 'w') as fout:
        for gidx in range(N):
            tok = tokendict[gidx]
            line = str(tok.sidx) + "\t" + str(tok.tidx) + "\t"
            line += tok.word + "\t" + tok.lemma + "\t"
            line += tok.pos + "\t" + tok.deplabel + "\t"
            line += str(tok.hidx) + "\t" + tok.ner + "\t"
            # 为了程序能继续走下去 修改了代码 张龙印 log
            if tok.partialparse is None:
                line += 'None' + "\t" + str(eduidx) + "\n"
            else:
                line += tok.partialparse + "\t" + str(eduidx) + "\n"
            # print("----->",type(tok.partialparse),'  eduindex-->',eduidx)
            fout.write(line)
            # Boundary
            if tok.boundary:
                eduidx += 1
            if tok.send:
                fout.write("\n")


def main(fmodel, fvocab, rpath, wpath):
    """
    from sklearn import svm
    :param fmodel:
    :param fvocab:
    :param rpath:
    :param wpath:
    :return:
    """
    global rpath_, wpath_, clf_, vocab_, dr_
    # extract
    extract(rpath)
    # seg
    rpath_ = rpath
    wpath_ = wpath
    clf_.loadmodel(fmodel)
    vocab_ = cPickle.load(gzip.open(fvocab))
    dr_ = DocReader()
    flist = [join(rpath, fname) for fname in listdir(rpath) if fname.endswith('conll')]
    pool = mp.Pool(processes=4)
    pool.map(do_seg_one_, flist)

