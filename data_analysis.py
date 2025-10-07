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

import pandas as pd
import os

def clean_and_save_data(greek_path, valiga_path, out_dir):
    """
    Funzione per pulire i dataset Greek e Valiga e salvare i file intermedi e finali.
    """
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

    # Selezione delle colonne rilevanti
    greek_relevant_cols = ['VALIGA_CODE','dateofbirth', 'RBdate', 'Lastvisit','Gender1M2F', 'age',  'uprot_gday', 'M', 'E', 'S', 'T', 'C', 'DiastolicbloodpressuremmHg', 'SystolicbloodpressuremmHg', 'AceiYN', 'Fishoil', 'CELLCEPT', 'AZA', 'CsIm', 'ESKDcorretto']
    greek = greek[greek_relevant_cols]
    valiga_relevant_cols = ['VALIGA CODE', 'SEX', 'M', 'E', 'S', 'T', 'C', 'dateAssess', 'systolic', 'Diastolic', 'age', 'outcome', 'Uprot', 'Nb of Bpmeds', 'RAS blockers', 'fish oil', 'Immunotherapies']
    valiga = valiga[valiga_relevant_cols]

    # Creazione della colonna dateAssess per Greek
    greek['dateAssess'] = (pd.to_datetime(greek['Lastvisit'], format="%d/%m/%Y") - 
                           pd.to_datetime(greek['RBdate'], format="%d/%m/%Y")).dt.days

    # Funzioni di processing
    def process_gender(df, gender_col):
        df['Gender'] = df[gender_col].map({1: 'M', 2: 'F', 'M': 'M', 'F': 'F'})
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
        # Se il valore di proteinuria è mancante, lo setto con il valore della visita precedente
        df['Proteinuria'] = df['Proteinuria'].fillna(method='ffill')
        return df

    def process_therapy_greek(df):
        df['Antihypertensive'] = df['Antihypertensive'].map({'Y': 1, 'N': 0, 1: 1, 0: 0}).astype('Int64')
        df['Immunosuppressants'] = (
            df['CsIm'].map({'Y': 1, 1: 1, 'N': 0, 0: 0}).astype('Int64') |
            df['AZA'].map({'Y': 1, 1: 1, 'N': 0, 0: 0}).astype('Int64') |
            df['CELLCEPT'].map({'Y': 1, 1: 1, 'N': 0, 0: 0}).astype('Int64')
        )
        df['FishOil'] = df['FishOil'].map({'Y': 1, 'N': 0, 1: 1, 0: 0}).astype('Int64')
        return df

    def process_therapy_valiga(df):
        df['Antihypertensive'] = (pd.to_numeric(df['Nb of Bpmeds'], errors='coerce') > 0).astype('Int64')
        df['Immunosuppressants'] = df['Immunotherapies'].map({'Y': 1, 'Yes': 1, 1: 1, 'N': 0, 'No': 0, 0: 0}).astype('Int64')
        df['FishOil'] = df['FishOil'].map({'Y': 1, 'Yes': 1, 1: 1, 'N': 0, 'No': 0, 0: 0}).astype('Int64')
        return df

    def process_eskd(df, eskd_col):
        df['Eskd'] = df[eskd_col].map({1: 1, '1': 1, 0: 0, '0': 0}).astype('Int64')
        return df

    # Rinomina colonne
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
        'Nb of Bpmeds': 'Nb of Bpmeds',
        'RAS blockers': 'RAS_blockers',
        'fish oil': 'FishOil',
        'Immunotherapies': 'Immunotherapies',
        'age': 'Age',
        'outcome': 'Eskd'
    }, inplace=True)

    # Processing
    greek = process_gender(greek, 'Gender')
    greek = process_hypertension(greek, 'Systolic', 'Diastolic', 'Antihypertensive')
    greek = process_proteinuria(greek, 'Proteinuria')
    greek = process_therapy_greek(greek)
    greek = process_eskd(greek, 'Eskd')

    valiga = process_gender(valiga, 'Gender')
    valiga = process_hypertension(valiga, 'Systolic', 'Diastolic')
    valiga = process_proteinuria(valiga, 'Proteinuria')
    valiga = process_therapy_valiga(valiga)
    valiga = process_eskd(valiga, 'Eskd')

    # Selezione colonne finali
    last_cols_to_keep = ['VALIGA_CODE','Gender', 'Age', 'dateAssess', 'Hypertension', 'M', 'E', 'S', 'T', 'C', 'Proteinuria', 'Antihypertensive', 'Immunosuppressants', 'FishOil', 'Eskd']
    greek = greek[last_cols_to_keep]
    valiga = valiga[last_cols_to_keep]

    # Salvataggio file puliti
    greek.to_excel(os.path.join(out_dir, 'cleaned_greek.xlsx'), index=False)
    valiga.to_excel(os.path.join(out_dir, 'cleaned_valiga.xlsx'), index=False)
    print("File salvati come cleaned_greek.xlsx e cleaned_valiga.xlsx")

    # Unione dei dataset
    combined = pd.concat([greek, valiga], ignore_index=True)
    combined['Age'] = combined['Age'].round().astype('Int64')
    combined.rename(columns={'VALIGA_CODE': 'Code'}, inplace=True)
    combined['Gender'] = combined['Gender'].map({1: 'M', 2: 'F', 'M': 'M', 'F': 'F'})

    # Salvataggio finale
    final = combined.loc[combined.groupby('Code')['dateAssess'].idxmax()].reset_index(drop=True)
    final.to_excel(os.path.join(out_dir, 'final_cleaned_maxDateAccess.xlsx'), index=False)
    print("File salvato come final_cleaned_maxDateAccess.xlsx")

    return os.path.join(out_dir, 'final_cleaned_maxDateAccess.xlsx')


def analyze_final_file(final_file, years_threshold=5):
    """
    Funzione per contare pazienti con dateAssess > years_threshold anni e ESKD=1.
    """
    final = pd.read_excel(final_file, engine='openpyxl')
    threshold_days = years_threshold * 365
    subset = final[final['dateAssess'] > threshold_days]
    eskd_count = subset['Eskd'].sum()
    print(f"Numero di pazienti con ESKD e dateAssess > {years_threshold} anni: {eskd_count}")
    return eskd_count

if __name__ == "__main__":

    greek_path = '/work/grana_far2023_fomo/ESKD/Data/greek.xls' 
    valiga_path = '/work/grana_far2023_fomo/ESKD/Data/valiga.xlsx' 
    out_greek_path = '/work/grana_far2023_fomo/ESKD/Data/cleaned_greek.xlsx' 
    out_valiga_path = '/work/grana_far2023_fomo/ESKD/Data/cleaned_valiga.xlsx'
    final_file = clean_and_save_data(greek_path, valiga_path, '/work/grana_far2023_fomo/ESKD/Data')
    final_file = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAccess.xlsx'
    # analyze_final_file(final_file, years_threshold=5)
    # analyze_final_file(final_file, years_threshold=10)
    # Voglio aprire il file finale e verificare di che tipo è il dato dateAssess
    # df = pd.read_excel(final_file, engine='openpyxl')
    # print(df['dateAssess'].dtype)
    # # Voglio vedere le prime righe del file
    # print(df.head())
    # # Numero di pazienti con ESKD e dateAssess > 5 anni: 115
    # Numero di pazienti con ESKD e dateAssess > 10 anni: 43
    
