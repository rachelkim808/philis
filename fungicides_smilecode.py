import pandas as pd

# Load the CSV file
file_path = 'chemicalpropertise_Rui.csv'
data = pd.read_csv(file_path)

# Define a list of fungicide compounds (you can expand this list as needed)
fungicides = [
    "Azoxystrobin", "Cyproconazole", "Epoxiconazole", "Fenbuconazole", "Fludioxonil",
    "Fluquinconazole", "Flutriafol", "Hexaconazole", "Kresoxim-methyl", "Myclobutanil",
    "Propiconazole", "Pyraclostrobin", "Tebuconazole", "Trifloxystrobin", "Triflumizole"
]

# Filter the data for fungicides
fungicide_data = data[data['Compound'].isin(fungicides)]

# Extract the SMILES codes
smiles_codes = fungicide_data[['Compound', 'SMILES']]

# Print the SMILES codes
print(smiles_codes)

# Optionally, save the results to a new CSV file
smiles_codes.to_csv('fungicide_smiles.csv', index=False)