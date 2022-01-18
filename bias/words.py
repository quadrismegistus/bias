from .imports import *

def get_fields():
    with open(PATH_FIELDS) as f:
        return json.load(f)




def get_biascontrasts(fieldd=None):
    if fieldd is None:
        fieldd=get_fields()
    return [
        {
            'contrast':'Woman-Man',
            'source':'VG',
            'pos':fieldd['VG.Human.Female'],
            'neg':fieldd['VG.Human.Male'],
        },
        {
            'contrast':'Woman',
            'source':'VG',
            'pos':fieldd['VG.Human.Female'],
            'neg':[],
        },
        {
            'contrast':'Man',
            'source':'VG',
            'pos':fieldd['VG.Human.Male'],
            'neg':[],
        },
        {
            'contrast':'Woman-Man',
            'source':'KW',
            'pos':['woman'],
            'neg':['man'],
        },
        {
            'contrast':'Woman',
            'source':'KW',
            'pos':['woman'],
            'neg':[],
        },
        {
            'contrast':'Man',
            'source':'KW',
            'pos':['man'],
            'neg':[],
        },
    ]


def get_words_df():
    df=pd.read_csv(PATH_WORDS,sep='\t').set_index('word')
    df['pos0']=[x[0] for x in df.pos]
    return df

def get_words_pos(pos):
    dfw=get_words_df()
    dfw=dfw[dfw.pos0.isin(pos)]
    return set([x for x in dfw.index if type(x)==str and x])
def get_content_words(): return get_words_pos(pos={'n','j','v'})
def get_nouns(): return get_words_pos(pos={'n'})




def get_jobs():
    with open(PATH_WORDS_JOBS) as f:
        o=set([
            x for x in f.read().lower().split('\n')
            if not ' ' in x
        ])
    o-={'tender','child','wielding','atmospheric','cleaning','painting','educational','welding'}
    return o


def get_animals():
    with open(PATH_WORDS_ANIMALS) as f:
        o=set([
            x for x in f.read().lower().split('\n')
            if not ' ' in x
        ])
    return o