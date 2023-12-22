#!/usr/bin/env python
# coding: utf-8

# # Import LIBs

# In[59]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime as dt
from plotly.offline import iplot, plot
from plotly.subplots import make_subplots


# # LOADING DATA

# In[3]:


df = pd.read_csv("netflix1.csv")
df.sample(10)


# In[7]:


#shape of data
df.shape


# In[8]:


df.info()


# In[10]:


#null check
df.isnull().sum()


# # Visualize

# In[11]:


count_types=df['type'].value_counts()
count_types


# In[19]:


# For viz: Ratio of Movies & TV shows

x=df.groupby(['type'])['type'].count()
y=len(df)
r=((x/y)).round(2)

mf_ratio = pd.DataFrame(r).T


# In[22]:


fig, ax = plt.subplots(1,1,figsize=(6.5, 2.5))

ax.barh(mf_ratio.index, mf_ratio['Movie'], 
        color='#b20710', alpha=0.9, label='Male')
ax.barh(mf_ratio.index, mf_ratio['TV Show'], left=mf_ratio['Movie'], 
        color='#221f1f', alpha=0.9, label='Female')

ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
#ax.set_yticklabels(mf_ratio.index, fontfamily='serif', fontsize=11)


# movie percentage
for i in mf_ratio.index:
    ax.annotate(f"{int(mf_ratio['Movie'][i]*100)}%", 
                   xy=(mf_ratio['Movie'][i]/2, i),
                   va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
                   color='white')

    ax.annotate("Movie", 
                   xy=(mf_ratio['Movie'][i]/2, -0.25),
                   va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
                   color='white')
    
    
for i in mf_ratio.index:
    ax.annotate(f"{int(mf_ratio['TV Show'][i]*100)}%", 
                   xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, i),
                   va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
                   color='white')
    ax.annotate("TV Show", 
                   xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, -0.25),
                   va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
                   color='white')






# Title & Subtitle
fig.text(0.125,1.03,'Movie & TV Show distribution', fontfamily='serif',fontsize=15, fontweight='bold')

for s in ['top', 'left', 'right', 'bottom']:
    ax.spines[s].set_visible(False)
    


#ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))

# Removing legend due to labelled plot
ax.legend().set_visible(False)
plt.show()


# # TOP DIRECTORS ON NETFLIX (by number of Movie)

# In[29]:


directors = df['director'].value_counts()
directors


# In[60]:


px.bar(directors[1:11],
      x=directors[1:11],
      y=directors[1:11].index,
      color = directors[1:11].index,
        color_discrete_sequence = colors,
      text_auto = True,
      labels = dict(x='Number of movies', y='Directors'),
      orientation= 'h'
      )


# # Country

# In[37]:


countries = df['country'].value_counts()[:10]
countries


# In[61]:


country_type = df.groupby(['country', 'type']).size().unstack(fill_value=0).reset_index()

country_type['Total'] = country_type['Movie'] + country_type['TV Show']

country_type = country_type[country_type['country'] != 'Not Given']

country_type = country_type.sort_values(by='Total', ascending=False)

colors = ['#B81D24', '#221F1F']

fig = px.bar(country_type.head(10), x='country', y=['Movie', 'TV Show'],
             labels={'value': 'Count', 'variable': 'Type'},
             title='Top 10 Countries and their Streamed Movies and TV Shows',
             barmode='group',  # This stacks the bars next to each other
             color_discrete_map={key: value for key, value in zip(['Movie', 'TV Show'], colors)})

fig.update_traces(marker=dict(line=dict(width=4)))

fig.show()


# Increasing contents by year

# In[62]:


iplot(px.line(shows_added_per_year,
    title='Number Of Shows Added per year',
    x = shows_added_per_year.index,
    y = shows_added_per_year,
    markers = True , line_shape='linear'
))


# # Ratings

# In[50]:


rating = df['rating'].value_counts()
rating.head(10)


# In[63]:


iplot(px.bar(rating,
    title="Shows Rating On Netflix",
    color = rating.index,
    orientation = 'h',
    height = 720,
    text_auto=True,
    labels =dict(index='Rating',value='Frequency'),
))


# # Num of season

# In[53]:


seasons = df[df['duration'].str.contains('Season')]
seasons_count = seasons['duration'].value_counts()
seasons_count


# In[64]:


iplot(px.bar(seasons_count,
    title="Season per TV Show",
    color = seasons_count.index,
    orientation = 'h',
    height = 720,
    text_auto=True,
    labels =dict(index='Seasons',value='Sum'),
))


# # GENRES

# In[55]:


categories = df['listed_in'].str.split(', ', expand=True)

categories = categories.melt(value_name='category').dropna()['category']

top_categories = categories.value_counts().head(10)

top_categories


# In[65]:


top_categories_df = pd.DataFrame({'Category': top_categories.index, 'Count': top_categories.values})

fig = px.bar(top_categories_df, x='Count', y='Category', orientation='h',
             title='Top 10 Popular Categories for Movies & TV Shows',
             labels={'Count': 'Number of Shows', 'Category': 'Category'},
             color=top_categories_df.index,
             text='Count')

fig.show()


# In[ ]:




