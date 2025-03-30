import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
"""
# Import data from CSV file
df = pd.read_csv('medical_examination.csv')

# Add an overweight column
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25 else 0)

# Normalize cholesterol and glucose data
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)


def draw_cat_plot():
    # Create DataFrame for cat plot using pd.melt()
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Rename columns for clarity
    df_cat = df_cat.rename(columns={'variable': 'feature', 'value': 'value'})

    # Create categorical plot using sns.catplot()
    cat_plot = sns.catplot(x="feature", hue="value", col="cardio", data=df_cat, kind="count")

    # Get figure for output
    fig = cat_plot.fig

    # Save figure (do not modify)
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # Clean data by filtering out incorrect rows and outliers
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Generate mask for upper triangle of heat map
    mask = np.triu(corr)

    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw heat map with sns.heatmap()
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, cmap='coolwarm', linewidths=0.5, ax=ax)
    
    fig.savefig('heatmap.png')
    return fig
    """


# Import data from CSV file
df = pd.read_csv('medical_examination.csv')

# Add an overweight column
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25 else 0)

# Normalize cholesterol and glucose data
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)

def draw_cat_plot():
    # Create DataFrame for cat plot using pd.melt()
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Rename columns for clarity
    df_cat = df_cat.rename(columns={'variable': 'feature', 'value': 'value'})

    # Create categorical plot using sns.catplot()
    cat_plot = sns.catplot(x="feature", hue="value", col="cardio", data=df_cat, kind="count")

    # Get figure for output
    fig = cat_plot.fig

    # Save figure (do not modify)
    fig.savefig('catplot.png')
    return fig

def draw_heat_map():
    # Clean data by filtering out incorrect rows and outliers
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Generate mask for upper triangle of heat map
    mask = np.triu(corr)

    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw heat map with sns.heatmap()
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, cmap='coolwarm', linewidths=0.5, ax=ax)

    # Save figure (do not modify)
    fig.savefig('heatmap.png')
    return fig


























"""

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] =  df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25 else 0)

# Step 3: Normalize cholesterol and glucose data
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)
"""
"""
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)


# 4
# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1,
# make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 5
def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat = df_cat.rename(columns={'variable': 'feature', 'value': 'value'})
    cat_plot = sns.catplot(x="feature", hue="value", col="cardio", data=df_cat, kind="count")
    fig = cat_plot.fig
    

    # 5
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke',
    # 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename
    # one of the collumns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    df_cat = df_cat.rename(columns={0: 'total'})

    # Draw the catplot with 'sns.catplot()'
    graph = sns.catplot(data=df_cat, kind="bar", x="variable", y="total", hue="value", col="cardio")
    fig = graph.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig
  """  

"""
    # 6 as above

    
    df_cat = 
    

    # 7



    # 8
    fig = None
"""
"""
    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))
                 ]

    # 11
    
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))
                 ]

    # 12
    
     # Calculate the correlation matrix
    corr = df_heat.corr()
    

    # 13
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    



    # 14
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 9))
    

    # 15
    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, square=True, linewidths=0.5, annot=True, fmt="0.1f")

    # 16
    fig.savefig('heatmap.png')
    return fig
  """  
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
# Add 'overweight' column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1,
# make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke',
    # 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename
    # one of the collumns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    df_cat = df_cat.rename(columns={0: 'total'})

    # Draw the catplot with 'sns.catplot()'
    graph = sns.catplot(data=df_cat, kind="bar", x="variable", y="total", hue="value", col="cardio")
    fig = graph.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))
                 ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, square=True, linewidths=0.5, annot=True, fmt="0.1f")

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig

    """
