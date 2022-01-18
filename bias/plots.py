from .imports import *


def plot_extremes(df, x=KEY, y='random', title='', **kwargs):
    p9.options.figure_size=5,8
    figdf = pd.concat([
        df.head(25),
        df.tail(25),
    ]).sort_values(x,ascending=False)
    figdf['x_abs']=figdf[x].apply(abs)
    # figdf=figdf[figdf.x_abs>=1]
    figdf['word'] = pd.Categorical(figdf['word'], categories=figdf['word'])
    figdf['direction'] = figdf[x].apply(lambda y: 'Woman' if y<0 else 'Man')
    fig = p9.ggplot(figdf, p9.aes(x='word',y=x,label='word',fill='direction'))
    fig += p9.geom_col()
    fig += p9.coord_flip()
    fig += p9.theme_minimal()
    fig += p9.scale_fill_brewer(type='qual',palette=1)
    fig += p9.labs(title=title)
    fig += p9.theme(title=p9.element_text(size=11))
    if title:
        ofn=os.path.join(PATH_FIGS, f'fig.{title.replace(" ","_")}.png')
        fig.save(ofn)
    return fig


def plot_hist(df, words, key=KEY):
    p9.options.figure_size=11,8
    figdf=df[df.word.isin(set(words))]
    figdf=pd.concat([g.set_index(['word','period']).sort_index().rolling(2,min_periods=0).mean().reset_index() for i,g in figdf.groupby('word')])
    figdf['period_int'] = figdf['period'].apply(lambda x: int(x[:4]))
    fig = p9.ggplot(figdf, p9.aes(x='period_int', y=key, color='word', label='word'))
    fig+=p9.geom_hline(yintercept=0, size=.25)
    fig +=p9.geom_line(size=.75)
    fig +=p9.geom_point()
    fig+=p9.theme_minimal()
    fig+=p9.geom_label(data=figdf[figdf.period_int==figdf.period_int.max()], size=10)
    # fig+=p9.geom_smooth(method='lm')
    fig+=p9.scale_color_brewer(type='qual',palette=2)
    return fig



def start_fig(data=None, theme='minimal',text_size=8, figure_size=(8,8), **aesd):
    p9.options.figure_size=figure_size
    p9.options.dpi=300
    fig=p9.ggplot(p9.aes(**aesd), data=data)
    fig+=getattr(p9,f'theme_{theme}')()
    fig+=p9.theme(
        text=p9.element_text(size=text_size),
    )
    return fig



def plot_neighbors(w, dfneighb=None, max_rank1=5, max_rank2=10, min_periods=2, width=9, height=6, color_by=KEY):
    wdf=find_most_similar(w)
    wdf=wdf[wdf['rank']<=max_rank2]
    
    words_ever_high=set(wdf[wdf['rank']<=max_rank1].word)
    figdf=wdf[wdf.word.isin(words_ever_high)].groupby(
        ['period','word']
    ).mean().reset_index()
    figdf=figdf.groupby('word').filter(lambda gdf: gdf.dropna().period.nunique()>=min_periods)
    figdf=figdf.set_index(['word','period'])
    if dfneighb is not None:
        figdf[color_by]=dfneighb[color_by]
        figdf=figdf[~figdf[color_by].isna()]
    figdf=figdf.reset_index()
    figdf=pd.concat(
        grp.assign(rank2=(grp['rank']).rank(method='first'))
        for i,grp in figdf.groupby('period')
    )
    figdf=figdf.sort_values(['period','rank'])
    print(f'Number of unique neighbors: {len(set(figdf.word))}')
    # start fig
    fig=start_fig(
        figdf,
        x='period',
        y='rank2',
        label='word',
        fill=color_by,
        figure_size=(width,height)
    ) + p9.geom_line(p9.aes(group='word'),size=0.5) + p9.scale_y_reverse()

    fig+=p9.geom_label(size=8, lineheight=2)
    fig+=p9.labs(
        title=f'Most similar words to "{w}"',
        x='Date of semantic model',
        y=f'Proximity to {w} →',
        fill=color_by
    )
    fig+=p9.theme(text=p9.element_text(size=10))
    fig+=p9.scale_fill_distiller(type='div', palette=2)
    return fig