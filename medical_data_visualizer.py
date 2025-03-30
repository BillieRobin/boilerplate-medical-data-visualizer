import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
# 1
df = None

# 2
df['overweight'] = None

# 3


# 4
def draw_cat_plot():
    # 5
    df_cat = None


    # 6
    df_cat = None
    

    # 7



    # 8
    fig = None


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = None

    # 12
    corr = None

    # 13
    mask = None



    # 14
    fig, ax = None

    # 15
"""

"""
    # 16
    fig.savefig('heatmap.png')
    return fig
"""
