"""
1. As first step, the dataset must be cleaned:
 The columns that must be mainteined are : Age, Sex, Hypertension (derivable from systolic and diastolic), MEST-C, Proteinuria and Therapy
 For what concern Hypertension: following paper hints, if systolic >= 140 mm Hg OR diastolic >= 90 mm Hg or patients receive antihypertensive drugs we can set the value of Hypertension to 1, 0 instead.
 For what concern Proteinuria, in the input files are reported proteinuria values starting from the first then to the last visit, we can simply use the first one registered during the RBdate (renal biopsy date).
 For what concern Therapy, we have to understand the type of treatment that have the patient. Specifically, we can map the different features in the 2 excels to some common features name.
 
 The mapping structure is the following:
    Farmaci antipertensivi	: (Nb of Bpmeds / RAS blockers - AceiYN)
    Immunosoppressori	:  (Immunotherapies	CsIm / AZA / CELLCEPT)
    Supplementi (olio di pesce)	: (fish oil - Fishoil)
    Intervento chirurgico	Tonsillectomy	—

There is just one column that can't be mapped beacuse it is absent in one of the Greek.xlsx file, that is the Tonsillectomy feature, it indicates if the patient encountered that specific medical surgery.

"""