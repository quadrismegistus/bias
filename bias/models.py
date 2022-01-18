from .imports import *

YEARBIN=5

MODEL_CACHE2={}

def load_model(path_model,path_vocab=None,min_count=None,cache_bin=True,cache=True):
    global MODEL_CACHE2
    if cache and path_model in MODEL_CACHE2: return MODEL_CACHE2[path_model]
    path,model=do_load_model(path_model,path_vocab=path_vocab,min_count=min_count,cache_bin=cache_bin)
    return model
        
    
def do_load_model(path_model,path_vocab=None,min_count=None,cache_bin=True):
#     print('>> loading',path_model)
    path_model_bin=path_model.split('.txt')[0]+'.bin' if not path_model.endswith('.bin') else path_model
    if os.path.exists(path_model_bin):
        model=gensim.models.KeyedVectors.load(path_model_bin,mmap='r')
    elif os.path.exists(path_model):
        if not path_vocab: path_vocab=os.path.join(os.path.dirname(path_model),'vocab.txt')
        if os.path.exists(path_vocab):
            model = gensim.models.KeyedVectors.load_word2vec_format(path_model,path_vocab)
            if min_count: filter_model(model,min_count=min_count)
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(path_model)
        if cache_bin:
            model.save(path_model_bin)
    else:
        print('!!??',path_model)
        stop
        return None,None
#     print(path_model, len(model.wv.key_to_index))
    return (path_model,model)
    
        
def get_path_model(corpus,period,run):
    return os.path.join(PATH_MODELS,corpus,period,f'run_{run:02}','model.bin')

def get_model_paths(model_dir=PATH_MODELS,model_fn='model.bin',vocab_fn='vocab.txt',period_len=None,progress=False):
    """
    Get all models' paths
    """
    ld=[]
    for root,dirs,fns in tqdm(os.walk(model_dir),desc='Scanning directory for models',disable=not progress):
        if model_fn in set(fns):
            corpus,period,run=root.split('/')[-3:]
            if corpus!='coha': continue
            if not 'run_' in run:
                corpus,period=root.split('/')[-2:]
                run=None
            dx={
                'corpus':corpus,
                'period':period,
                'period_start':int(period.split('-')[0]),
                'period_end':int(period.split('-')[-1]),
                'path':os.path.join(root,model_fn),
                'path_vocab':os.path.join(root,vocab_fn)
            }
            if run is not None: dx['run']=run
            if period_len and int(dx['period_end'])-int(dx['period_start'])!=period_len:
                continue
            ld.append(dx)
    return ld

def get_pathdf_models(period_len=YEARBIN,ymin=YMIN,ymax=YMAX):
    pathdf=pd.DataFrame(get_model_paths(PATH_MODELS, 'model.bin'))#.sort_values(['period_start','run']
    return pathdf.sort_values(['corpus','period','run'])


def iter_models(pathdf=None):
    if pathdf is None: pathdf=get_pathdf_models()
    for i,row in pathdf.iterrows():
        m = load_model(row.path)
        for k,v in row.items():
            setattr(m,k,v)
        m.attrs=dict(row)
        yield m





def get_centroid(model,words):
    words=[words] if type(words)==str else list(words)
    vectors=[]
    for w in words:
        if w in model.key_to_index:
            vectors+=[model[w]]
    if not vectors: return None
    return np.mean(vectors,0)

def compute_vector(model,words_pos=[],words_neg=[]):
    centroid_pos=get_centroid(model,words_pos)
    if not words_neg: return centroid_pos
    centroid_neg=get_centroid(model,words_neg)
    if centroid_neg is not None:
        return centroid_pos - centroid_neg
    else:
        return centroid_pos


def do_compute_vector_scores(obj):
    vec1,vec2=obj
    return distance.cosine(vec1,vec2)

def compute_vector_scores(m,pos,neg=None,z=True):
    vec=np.array(compute_vector(m,pos,neg), dtype=np.float64)
    matrix=np.array(m.get_normed_vectors(),dtype=np.float64)
    objs=[(vec,vec2) for vec2 in matrix]
    res = map(do_compute_vector_scores,objs)
    resd=dict((m.index_to_key[i],x)for i,x in enumerate(res))
    s=pd.Series(resd)
    if z: s=(s - s.mean())/s.std()
    return s.sort_values()


def compute_key_vector_scores(m,contrasts=None):
    # get fields
    if contrasts is None:
        from .words import get_biascontrasts
        contrasts=get_biascontrasts()
    odf=pd.DataFrame()
    for cd in contrasts:
        odf[cd['contrast']+'.'+cd['source']] = compute_vector_scores(m, cd.get('pos',{}), cd.get('neg',{}))
    odf=odf.rename_axis('word').reset_index()
    return odf.set_index('word')


def do_compute_all_vectors(df,words=None,add_ambig=True, add_freq=True, add_sing=True):
    m=load_model(df.iloc[0].path)
    odf=compute_key_vector_scores(m)
    return odf.reset_index()

def compute_key_vector_scores_across_models(pathdf,words=None,num_proc=1,**attrs):
    #gby=[c for c in pathdf.columns if not c.startswith('path')]
    gby=['corpus','period','run']
    return pmap_groups(
        do_compute_all_vectors,
        pathdf.groupby(gby),
        num_proc=num_proc,
        kwargs=dict(words=words),
        desc='Computing key vectors across model runs'
    )



def measure_change(dfw, ykey=KEY):
    y=dfw[ykey]
    lm = sm.OLS(y, dfw['period_int'])
    res = lm.fit()
    
    dfw1=dfw[dfw.period_int == dfw.period_int.min()]
    dfw2=dfw[dfw.period_int == dfw.period_int.max()]
    start = dfw1[ykey].mean()
    end = dfw2[ykey].mean()

    def flipped(yx1=1, yx0=.05):
        if start<-yx1 and end>-yx0: return 'M'
        if start>yx1 and end<yx0: return 'F'
        return ''

    d=dict(
        r2=res.rsquared,
        n=res.nobs,
        p=res.pvalues[0],
        slope=res.params[0],
        min=y.min(),
        max=y.max(),
        magn=y.max() - y.min(),
        start=start,
        end=end,
        change=end-start,
        flipped=flipped()
    )
    odf=pd.DataFrame([d])
    return odf

def measure_changing_words(dfch=None, ykey=KEY):
    if dfch is None: dfch = load_data(avg_runs=False)
    dfch = dfch[dfch.num_models==dfch.num_models.max()].sort_values(['word','period'])
    dfch_words = pmap_groups(measure_change, dfch.groupby('word'), num_proc=8, kwargs=dict(ykey=ykey))
    dfch_words = dfch_words[dfch_words.p<=0.05]
    dfch_words = dfch_words.sort_values('r2',ascending=False)
    return dfch_words



def find_most_similar(words, ok_words=None):
    o=[]
    for m in tqdm(iter_models()):
        if m.corpus!='coha': continue
        wi2=0
        for wi,(w,csim) in enumerate(m.most_similar(words,topn=25)):
            if ok_words and not w in ok_words: continue
            wi2+=1
            odx={'word':w, 'csim':csim, 'rank':wi2, **m.attrs}
            o.append(odx)
    odf=pd.DataFrame(o).drop(['path','path_vocab'],1)
    
    o=[]
    for i,g in odf.groupby(['corpus','period']):
        s=g.word.value_counts()
        g=g.set_index('word')
        g['num_nbr']=s
        g=g.reset_index()
        ok=s[s>1].index
        g2=g[g.word.isin(ok)]
        o.append(g2)
    dfsim2 = pd.concat(o)
    return dfsim2



def load_bias_data(min_count=10, pos={'n','j','v'}, words=None, avg_runs=True):
    df = pd.read_pickle(PATH_WM_VEC_SCORES).reset_index()
    df = df[df.corpus=='coha']
    s1=df.word.value_counts()
    df=df.set_index('word')
    df['num_models']=s1
    df=df.reset_index()

    s=s1 >= min_count
    ok_words=set(s[s].index)
    df = df[df.word.isin(ok_words)]

    if avg_runs:
        df=df.groupby(['corpus','period','word']).mean().reset_index()
        s2=df.word.value_counts()
        df=df.set_index('word')
        df['num_periods']=s2
        df=df.reset_index()

    posdf=get_words_df()
    posd=dict(zip(posdf.index, posdf.pos0))
    df['pos']=[posd.get(x,'') for x in df.word]
    df=df[df.pos.isin(pos)]
    df['period_int']=df.period.apply(lambda x: int(x[:4]))
    
    return df.sort_values('Woman-Man.VG')