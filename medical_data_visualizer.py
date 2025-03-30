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

    # Rename columns for clarity (expected name is 'variable')
    df_cat = df_cat.rename(columns={'variable': 'variable', 'value': 'value'})

    # Define the correct order for x-axis labels
    variable_order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']

    # Create categorical plot using sns.catplot()
    cat_plot = sns.catplot(x="variable", hue="value", col="cardio", data=df_cat, kind="count", order=variable_order)

    # Set ylabel to 'total' as expected by the test
    cat_plot.set_axis_labels("variable", "total")

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

    # Drop unnecessary columns ('bmi') for heatmap calculations
    df_heat = df_heat.drop(columns=['bmi'])

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

