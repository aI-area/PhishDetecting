import pandas as pd

# Load dataset
df = pd.read_csv("phishing_site_urls.csv")   # change filename if needed

# Check label distribution (optional but recommended)
print(df['phishing'].value_counts())

# Sample 100,000 legitimate URLs (label = 0)
legitimate = df[df['phishing'] == 0].sample(n=100000, random_state=42)

# Sample 100,000 phishing URLs (label = 1)
phishing = df[df['phishing'] == 1].sample(n=100000, random_state=42)

# Combine both
balanced_df = pd.concat([legitimate, phishing])

# Shuffle the final dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new file
balanced_df.to_csv("balanced_200k_urls.csv", index=False)

print("Balanced dataset created:", balanced_df.shape)
