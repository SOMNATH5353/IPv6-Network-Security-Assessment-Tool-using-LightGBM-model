import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def optimize_memory(df):
    # Convert object types to category to save memory
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

def generate_vulnerability_graph(input_csv, output_image):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    print(f"Reading data from {input_csv}")
    data = pd.read_csv(input_csv)
    
    # Optimize memory usage
    data = optimize_memory(data)
    
    # Check for required columns
    required_columns = ['vulnerability_name', 'predicted_vulnerability']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"The column '{col}' is missing from the dataset.")
    
    # Limit to top N vulnerabilities for readability
    top_n = 20
    vulnerability_counts = data['vulnerability_name'].value_counts().reset_index()
    vulnerability_counts.columns = ['vulnerability_name', 'count']
    top_vulnerabilities = vulnerability_counts.head(top_n)['vulnerability_name']
    data_top = data[data['vulnerability_name'].isin(top_vulnerabilities)]
    
    # Recalculate counts for top vulnerabilities
    vulnerability_counts_top = data_top['vulnerability_name'].value_counts().reset_index()
    vulnerability_counts_top.columns = ['vulnerability_name', 'count']
    
    # Count the occurrences of each vulnerability and their variations
    variation_counts = data_top.groupby(['vulnerability_name', 'predicted_vulnerability']).size().reset_index(name='count')
    
    print("Generating vulnerability graph...")
    
    # Set the style for seaborn
    sns.set(style="whitegrid")
    
    # Plot 1: Bar Chart for Top Vulnerability Counts
    plt.figure(figsize=(14, 8))
    sns.barplot(data=vulnerability_counts_top, x='vulnerability_name', y='count', palette='viridis')
    plt.title('Top Detected Vulnerabilities', fontsize=18)
    plt.xlabel('Vulnerability Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Top vulnerability count bar chart saved as {output_image}")
    plt.show()
    
    # Plot 2: Count Plot with Variations (Grouped Bar Chart)
    plt.figure(figsize=(14, 8))
    sns.countplot(data=data_top, x='vulnerability_name', hue='vulnerability_variation', palette='viridis')
    plt.title('Top Detected Vulnerabilities with Variations', fontsize=18)
    plt.xlabel('Vulnerability Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Variation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    variation_output_image = output_image.replace('.png', '_with_variations.png')
    plt.savefig(variation_output_image)
    print(f"Vulnerability variations count plot saved as {variation_output_image}")
    plt.show()
    
    # Plot 3: Stacked Bar Chart for Vulnerability Variations
    pivot_table = variation_counts.pivot(index='vulnerability_name', columns='vulnerability_variation', values='count').fillna(0)
    
    pivot_table.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
    plt.title('Top Detected Vulnerabilities with Variations (Stacked)', fontsize=18)
    plt.xlabel('Vulnerability Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Variation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    stacked_output_image = output_image.replace('.png', '_stacked.png')
    plt.savefig(stacked_output_image)
    print(f"Vulnerability variations stacked bar chart saved as {stacked_output_image}")
    plt.show()

if __name__ == "__main__":
    generate_vulnerability_graph('data/final_output.csv', 'images/vulnerability_graph.png')
