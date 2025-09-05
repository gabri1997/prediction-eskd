"""
1. As first step, the dataset must be cleaned:
 The columns that must be mainteined are : Age, Gender, Hypertension (derivable from systolic and diastolic), MEST-C, Proteinuria and Therapy
 For what concern Hypertension: following paper hints, if systolic >= 140 mm Hg OR diastolic >= 90 mm Hg or patients receive antihypertensive drugs we can set the value of Hypertension to 1, 0 instead.
 For what concern Proteinuria, in the input files are reported proteinuria values starting from the first then to the last visit, we can simply use the first one registered during the RBdate (renal biopsy date).
 For what concern Therapy, we have to understand the type of treatment that have the patient. Specifically, we can map the different features in the 2 excels to some common features name.
 
 The mapping structure is the following:
    Farmaci antipertensivi	: (Nb of Bpmeds / RAS blockers - AceiYN)
    Immunosoppressori	:  (Immunotherapies	CsIm / AZA / CELLCEPT)
    Supplementi (olio di pesce)	: (fish oil - Fishoil)
    Intervento chirurgico	Tonsillectomy	—

There is just one column that can't be mapped because it is absent in one of the Greek.xlsx file, that is the Tonsillectomy feature, it indicates if the patient encountered that specific medical surgery.

Then I would like to have something like:

Code|Gender|Age|AssesDate|Hypertesnion|M|E|S|T|C|Proteinuria|Antihypertensive|Immunosuppressants|FishOil

where Antihypertensive|Immunosuppressants|FishOil are the Therapy.

In Greek file have been added the column dateAssess that is the difference in days between Lastvisit and RBdate.

"""

# Pulizia di Greek
import pandas as pd
import os

greek_path = '/work/grana_far2023_fomo/ESKD/Data/greek.xls'
valiga_path = '/work/grana_far2023_fomo/ESKD/Data/valiga.xlsx'
out_greek_path = '/work/grana_far2023_fomo/ESKD/Data/cleaned_greek.xlsx'
out_valiga_path = '/work/grana_far2023_fomo/ESKD/Data/cleaned_valiga.xlsx'

def func(x):
    dl = x.split()[0].split('-')
    return f"{dl[1]}/{dl[2]}/{dl[0]}"

def read_excel_auto(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".xls":
        return pd.read_excel(path, engine="xlrd", parse_dates=True)
    elif ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl", parse_dates=True)
    else:
        raise ValueError(f"Formato non supportato: {ext}")

greek = read_excel_auto(greek_path)
valiga = read_excel_auto(valiga_path)

print("Colonne Greek:", greek.columns.tolist())
print("Colonne Valiga:", valiga.columns.tolist())

# Selezione delle colonne rilevanti
greek_relevant_cols = ['dateofbirth', 'RBdate', 'Lastvisit','Gender1M2F', 'age',  'uprot_gday', 'M', 'E', 'S', 'T', 'C', 'DiastolicbloodpressuremmHg', 'SystolicbloodpressuremmHg', 'AceiYN', 'Fishoil', 'CELLCEPT', 'AZA', 'CsIm']
greek = greek[greek_relevant_cols]
valiga_relevant_cols = ['VALIGA CODE', 'SEX', 'M', 'E', 'S', 'T', 'C', 'dateAssess', 'systolic', 'Diastolic', 'age', 'outcome', 'Uprot', 'Nb of Bpmeds', 'RAS blockers', 'fish oil', 'Immunotherapies']
valiga = valiga[valiga_relevant_cols]

# Voglio creare una nuova colonna in greek chiamata dateAsses che corrisponde alla differenza in giorni tra Lastvisit e RBdate
greek['dateAssess'] = (pd.to_datetime(greek['Lastvisit'], format="%d/%m/%Y") - pd.to_datetime(greek['RBdate'], format="%d/%m/%Y")).dt.days

print(greek['dateAssess'].head())

# Colonne di tipo data in Greek
date_cols_greek = ['dateofbirth', 'RBdate', 'Lastvisit']
for col in date_cols_greek:
    greek[col] = pd.to_datetime(greek[col], errors='coerce').dt.strftime('%d/%m/%Y')

if os.path.exists(out_greek_path):
    print(f"Il file {out_greek_path} esiste già. Non verrà sovrascritto.")
else:
    greek.to_excel('/work/grana_far2023_fomo/ESKD/Data/cleaned_greek.xlsx', index=False)
    print(f"File salvato come {out_greek_path}")

if os.path.exists('/work/grana_far2023_fomo/ESKD/Data/cleaned_valiga.xlsx'):
    print("Il file cleaned_valiga.xlsx esiste già. Non verrà sovrascritto.")
else:
    valiga.to_excel('/work/grana_far2023_fomo/ESKD/Data/cleaned_valiga.xlsx', index=False)
    print("File salvato come cleaned_valiga.xlsx")

print('Colonne greek: ',greek.columns)
print('Colonne valiga: ',valiga.columns)
 
# Devo uniformare le colonne
final_cols = ['Interval', 'Gender', 'Age', 'M', 'E', 'S', 'T', 'C', 'Proteinuria', 'Hypertension', 'Antihypertensive', 'Immunosuppressants', 'FishOil', 'Eskd']

