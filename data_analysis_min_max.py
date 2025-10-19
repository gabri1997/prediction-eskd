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

def clean_and_save_data(greek_path, valiga_path, out_dir, row_to_keep='min'):
    """
    Funzione per pulire i dataset Greek e Valiga e salvare i file intermedi e finali.
    
    Parameters:
    -----------
    row_to_keep : str
        'min' = usa dati BASELINE (prima visita/biopsia) - per predizione precoce
        'max' = usa dati LAST (ultima visita) - per analisi retrospettiva
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

    print(f"\n{'='*60}")
    print(f"Modalità selezionata: {row_to_keep.upper()}")
    if row_to_keep == 'min':
        print("Usando dati BASELINE (prima visita/biopsia)")
    else:
        print("Usando dati ULTIMA VISITA")
    print(f"{'='*60}\n")

    # Selezione delle colonne rilevanti in base a row_to_keep
    if row_to_keep == 'min':
        # BASELINE: creat_mgdl, uprot_gday, Systolic/Diastolic normali
        greek_relevant_cols = [
            'VALIGA_CODE', 'dateofbirth', 'RBdate', 'Lastvisit', 'Gender1M2F', 'age',
            'creat_mgdl',  # BASELINE creatinina
            'uprot_gday',  # BASELINE proteinuria
            'M', 'E', 'S', 'T', 'C',
            'DiastolicbloodpressuremmHg',  # BASELINE
            'SystolicbloodpressuremmHg',   # BASELINE
            'AceiYN', 'Fishoil', 'CELLCEPT', 'AZA', 'CsIm', 'ESKDcorretto'
        ]
    else:
        # LAST: creat_mgdl_last, uprot_gday_last, SBPlast, DBPlast
        greek_relevant_cols = [
            'VALIGA_CODE', 'dateofbirth', 'RBdate', 'Lastvisit', 'Gender1M2F', 'age',
            'creat_mgdl_last',  # LAST creatinina
            'uprot_gday_last',  # LAST proteinuria
            'M', 'E', 'S', 'T', 'C',
            'DBPlast',  # LAST diastolica
            'SBPlast',  # LAST sistolica
            'AceiYN', 'Fishoil', 'CELLCEPT', 'AZA', 'CsIm', 'ESKDcorretto'
        ]
    
    greek = greek[greek_relevant_cols]
    
    # Valiga (assumo abbia sempre dati baseline, modifica se necessario)
    valiga_relevant_cols = [
        'VALIGA CODE', 'SEX', 'M', 'E', 'S', 'T', 'C', 'dateAssess',
        'systolic', 'Diastolic', 'age', 'creat_mg_dl', 'outcome', 'Uprot',
        'Nb of Bpmeds', 'RAS blockers', 'fish oil', 'Immunotherapies'
    ]
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
        df['Proteinuria'] = df['Proteinuria'].fillna(method='ffill')
        return df

    def process_creatinine(df, creat_col):
        df['Creatinine'] = pd.to_numeric(df[creat_col], errors='coerce')
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

    # Rinomina colonne GREEK (dipende da row_to_keep)
    if row_to_keep == 'min':
        greek.rename(columns={
            'Gender1M2F': 'Gender',
            'creat_mgdl': 'Creatinine_raw',  # BASELINE
            'uprot_gday': 'Proteinuria',     # BASELINE
            'DiastolicbloodpressuremmHg': 'Diastolic',
            'SystolicbloodpressuremmHg': 'Systolic',
            'AceiYN': 'Antihypertensive',
            'Fishoil': 'FishOil',
            'ESKDcorretto': 'Eskd',
            'age': 'Age'
        }, inplace=True)
    else:
        greek.rename(columns={
            'Gender1M2F': 'Gender',
            'creat_mgdl_last': 'Creatinine_raw',  # LAST
            'uprot_gday_last': 'Proteinuria',     # LAST
            'DBPlast': 'Diastolic',
            'SBPlast': 'Systolic',
            'AceiYN': 'Antihypertensive',
            'Fishoil': 'FishOil',
            'ESKDcorretto': 'Eskd',
            'age': 'Age'
        }, inplace=True)

    # Rinomina colonne VALIGA
    valiga.rename(columns={
        'VALIGA CODE': 'VALIGA_CODE',
        'SEX': 'Gender',
        'Uprot': 'Proteinuria',
        'creat_mg_dl': 'Creatinine_raw',
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
    greek = process_creatinine(greek, 'Creatinine_raw')
    greek = process_therapy_greek(greek)
    greek = process_eskd(greek, 'Eskd')

    valiga = process_gender(valiga, 'Gender')
    valiga = process_hypertension(valiga, 'Systolic', 'Diastolic')
    valiga = process_proteinuria(valiga, 'Proteinuria')
    valiga = process_creatinine(valiga, 'Creatinine_raw')
    valiga = process_therapy_valiga(valiga)
    valiga = process_eskd(valiga, 'Eskd')

    # Selezione colonne finali
    last_cols_to_keep = [
        'VALIGA_CODE', 'Gender', 'Age', 'dateAssess', 'Hypertension',
        'M', 'E', 'S', 'T', 'C', 'Proteinuria', 'Creatinine',
        'Antihypertensive', 'Immunosuppressants', 'FishOil', 'Eskd'
    ]

    greek = greek[last_cols_to_keep]
    valiga = valiga[last_cols_to_keep]

    # Salvataggio file puliti
    suffix = "_baseline" if row_to_keep == 'min' else "_last"
    greek.to_excel(os.path.join(out_dir, f'cleaned_greek{suffix}.xlsx'), index=False)
    valiga.to_excel(os.path.join(out_dir, f'cleaned_valiga{suffix}.xlsx'), index=False)
    print(f"File salvati come cleaned_greek{suffix}.xlsx e cleaned_valiga{suffix}.xlsx")

    # Unione dei dataset
    combined = pd.concat([greek, valiga], ignore_index=True)
    combined['Age'] = combined['Age'].round().astype('Int64')
    combined.rename(columns={'VALIGA_CODE': 'Code'}, inplace=True)
    combined['Gender'] = combined['Gender'].map({1: 'M', 2: 'F', 'M': 'M', 'F': 'F'})

    # Calcolo del record finale per ogni paziente
    if row_to_keep == 'max':
        # Usa sempre la data maggiore (ultima visita)
        final = combined.loc[combined.groupby('Code')['dateAssess'].idxmax()].reset_index(drop=True)
        output_file = os.path.join(out_dir, 'final_cleaned_maxDateAccess.xlsx')
    else:
        # Per Greek: prendi la prima (min)
        greek_final = greek.loc[greek.groupby('VALIGA_CODE')['dateAssess'].idxmin()].reset_index(drop=True)
        # Per Valiga: mantieni comunque la data maggiore, tanto nella classificazione non viene usato e nella regressione mi serve il numero di giorni passati tra prima e ultima visita
        valiga_final = valiga.loc[valiga.groupby('VALIGA_CODE')['dateAssess'].idxmax()].reset_index(drop=True)
        # Combina i due dataset
        final = pd.concat([greek_final, valiga_final], ignore_index=True)
        final.rename(columns={'VALIGA_CODE': 'Code'}, inplace=True)
        output_file = os.path.join(out_dir, 'final_cleaned_minDateAccess.xlsx')

    
    final.to_excel(output_file, index=False)
    print(f"File salvato come {os.path.basename(output_file)}")
    
    # Statistiche finali
    print(f"\n{'='*60}")
    print(f"Statistiche dataset finale:")
    print(f"Totale pazienti: {len(final)}")
    print(f"Pazienti con ESKD: {final['Eskd'].sum()} ({final['Eskd'].sum()/len(final)*100:.1f}%)")
    print(f"Follow-up medio: {final['dateAssess'].mean()/365:.1f} anni")
    print(f"Range creatinina: [{final['Creatinine'].min():.2f}, {final['Creatinine'].max():.2f}] mg/dL")
    print(f"Range proteinuria: [{final['Proteinuria'].min():.2f}, {final['Proteinuria'].max():.2f}] g/day")
    print(f"{'='*60}\n")

    return output_file


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
    
    # IMPORTANTE: Scegli 'min' per BASELINE o 'max' per LAST
    row_to_keep = 'min'  # 'min' = BASELINE (raccomandato per predizione), 'max' = LAST
    
    final_file = clean_and_save_data(greek_path, valiga_path, 
                                      '/work/grana_far2023_fomo/ESKD/Data', 
                                      row_to_keep)
    
    # Analisi opzionale
    # analyze_final_file(final_file, years_threshold=5)
    # analyze_final_file(final_file, years_threshold=10)
    
    # # Verifica dati
    # df = pd.read_excel(final_file, engine='openpyxl')
    # print(f"\nColonne finali: {list(df.columns)}")
    # print(f"\nPrime righe:\n{df.head()}")