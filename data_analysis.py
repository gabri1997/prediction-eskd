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
greek_relevant_cols = ['VALIGA_CODE','dateofbirth', 'RBdate', 'Lastvisit','Gender1M2F', 'age',  'uprot_gday', 'M', 'E', 'S', 'T', 'C', 'DiastolicbloodpressuremmHg', 'SystolicbloodpressuremmHg', 'AceiYN', 'Fishoil', 'CELLCEPT', 'AZA', 'CsIm', 'ESKDcorretto']
greek = greek[greek_relevant_cols]
valiga_relevant_cols = ['VALIGA CODE', 'SEX', 'M', 'E', 'S', 'T', 'C', 'dateAssess', 'systolic', 'Diastolic', 'age', 'outcome', 'Uprot', 'Nb of Bpmeds', 'RAS blockers', 'fish oil', 'Immunotherapies']
valiga = valiga[valiga_relevant_cols]

# Voglio creare una nuova colonna in greek chiamata dateAsses che corrisponde alla differenza in giorni tra Lastvisit e RBdate
greek['dateAssess'] = (pd.to_datetime(greek['Lastvisit'], format="%d/%m/%Y") - pd.to_datetime(greek['RBdate'], format="%d/%m/%Y")).dt.days

#print(greek['dateAssess'].head())

# Devo uniformare le colonne
#final_cols = ['Interval', 'Gender', 'Age', 'M', 'E', 'S', 'T', 'C', 'Proteinuria', 'Hypertension', 'Antihypertensive', 'Immunosuppressants', 'FishOil', 'Eskd']

# Ci vogiono delle funzioni per processare le colonne
def process_gender(df, gender_col):
    df['gender'] = df[gender_col].map({1: 'M', 2: 'F', 'M': 'M', 'F': 'F'})
    return df
def process_hypertension(df, sys_col, dia_col, acei_col=None):
  
    df[sys_col] = pd.to_numeric(df[sys_col], errors='coerce')
    df[dia_col] = pd.to_numeric(df[dia_col], errors='coerce')
    
    hypert = (df[sys_col] >= 140) | (df[dia_col] >= 90)
    
    if acei_col:
        hypert = hypert | (df[acei_col] == 'Y')
    
    df['Hypertension'] = hypert.astype(int)
    return df

def process_proteinuria(df, prot_col):
    df['Proteinuria'] = pd.to_numeric(df[prot_col], errors='coerce')
    return df
def process_therapy_greek(df):
    df['Antihypertensive'] = (df['AceiYN'] == 'Y').astype(int)
    df['Immunosuppressants'] = ((df['CsIm'] == 'Y') | (df['AZA'] == 'Y') | (df['CELLCEPT'] == 'Y')).astype(int)
    df['FishOil'] = (df['Fishoil'] == 'Y').astype(int)
    return df
def process_therapy_valiga(df):
    df['Antihypertensive'] = (df['Nb of Bpmeds'] > 0).astype(int)
    df['Immunosuppressants'] = (df['Immunotherapies'] == 'Y').astype(int)
    df['FishOil'] = (df['fish oil'] == 'Y').astype(int)
    return df
def process_eskd(df, eskd_col):
    df['Eskd'] = (df[eskd_col] == 1).astype(int)
    return df   

# Processamento Greek
greek = process_gender(greek, 'Gender1M2F')
greek = process_hypertension(greek, 'SystolicbloodpressuremmHg', 'DiastolicbloodpressuremmHg', 'AceiYN')
greek = process_proteinuria(greek, 'uprot_gday')
greek = process_therapy_greek(greek)
greek['Eskd'] = greek['ESKDcorretto'] if 'ESKDcorretto' in greek.columns else 'Absent'
# Processamento Valiga
valiga = process_gender(valiga, 'SEX')
valiga = process_hypertension(valiga, 'systolic', 'Diastolic')
valiga = process_proteinuria(valiga, 'Uprot')
valiga = process_therapy_valiga(valiga)
valiga = process_eskd(valiga, 'outcome')

# Colonne di tipo data in Greek
date_cols_greek = ['dateofbirth', 'RBdate', 'Lastvisit']
for col in date_cols_greek:
    greek[col] = pd.to_datetime(greek[col], errors='coerce').dt.strftime('%d/%m/%Y')

greek.rename(columns={
    'Gender1M2F': 'Gender',
    'uprot_gday': 'Proteinuria',
    'DiastolicbloodpressuremmHg': 'Diastolic',
    'SystolicbloodpressuremmHg': 'Systolic',
    'AceiYN': 'Antihypertensive',
    'Fishoil': 'FishOil',
    'CELLCEPT': 'CELLCEPT',
    'AZA': 'AZA',
    'CsIm': 'CsIm',
    'ESKDcorretto': 'Eskd',
    'age': 'Age'
}, inplace=True)

valiga.rename(columns={
    'VALIGA CODE': 'VALIGA_CODE',
    'SEX': 'Gender',
    'Uprot': 'Proteinuria',
    'systolic': 'Systolic',
    'Diastolic': 'Diastolic',
    'Nb of Bpmeds': 'Nb_of_Bpmeds',
    'RAS blockers': 'RAS_blockers',
    'fish oil': 'FishOil',
    'Immunotherapies': 'Immunotherapies',
    'age': 'Age'
}, inplace=True)

last_cols_to_keep = ['VALIGA_CODE','Gender', 'Age', 'dateAssess', 'Hypertension', 'M', 'E', 'S', 'T', 'C', 'Proteinuria', 'Antihypertensive', 'Immunosuppressants', 'FishOil', 'Eskd']

greek = greek[last_cols_to_keep]
valiga = valiga[last_cols_to_keep]

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