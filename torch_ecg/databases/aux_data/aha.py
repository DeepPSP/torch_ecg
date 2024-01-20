"""
Standardized ECG diagnostic statements by AHA/ACC/HRS (i.e. the American Heart Association, the American College of Cardiology, and the Heart Rhythm Society)

References
----------
1. Mason, J. W., Hancock, E. W. & Gettes, L. S. Recommendations for the standardization and interpretation of the electrocardiogram.
Circulation 115, 1325–1332 (2007).
"""

import io

import pandas as pd

__all__ = [
    "df_primary_statements",
    "df_secondary_statements",
    "df_modifiers",
    "df_comparison_statements",
    "df_convenience_statements",
    "df_secondary_primary_statement_pairing_rules",
    "df_general_modifier_primary_statement_pairing_rules",
]


df_primary_statements = pd.read_csv(
    io.StringIO(
        """
CategoryCode,Category,Code,Description
A,Overall interpretation,1,Normal ECG
,,2,Otherwise normal ECG
,,3,Abnormal ECG
,,4,Uninterpretable ECG
B,Technical conditions,10,Extremity electrode reversal
,,11,Misplaced precordial electrode(s)
,,12,Missing lead(s)
,,13,Right-sided precordial electrode(s)
,,14,Artifact
,,15,Poor-quality data
,,16,Posterior electrode(s)
C,Sinus node rhythms and arrhythmias,20,Sinus rhythm
,,21,Sinus tachycardia
,,22,Sinus bradycardia
,,23,Sinus arrhythmia
,,24,"Sinoatrial block, type I"
,,25,"Sinoatrial block, type II"
,,26,Sinus pause or arrest
,,27,Uncertain supraventricular rhythm
D,Supraventricular arrhythmias,30,Atrial premature complex(es)
,,31,"Atrial premature complexes, nonconducted"
,,32,Retrograde atrial activation
,,33,Wandering atrial pacemaker
,,34,Ectopic atrial rhythm
,,35,"Ectopic atrial rhythm, multifocal"
,,36,Junctional premature complex(es)
,,37,Junctional escape complex(es)
,,38,Junctional rhythm
,,39,Accelerated junctional rhythm
,,40,Supraventricular rhythm
,,41,Supraventricular complex(es)
,,42,"Bradycardia, nonsinus"
E,Supraventricular tachyarrhythmias,50,Atrial fibrillation
,,51,Atrial flutter
,,52,"Ectopic atrial tachycardia, unifocal"
,,53,"Ectopic atrial tachycardia, multifocal"
,,54,Junctional tachycardia
,,55,Supraventricular tachycardia
,,56,Narrow-QRS tachycardia
F,Ventricular arrhythmias,60,Ventricular premature complex(es)
,,61,Fusion complex(es)
,,62,Ventricular escape complex(es)
,,63,Idioventricular rhythm
,,64,Accelerated idioventricular rhythm
,,65,Fascicular rhythm
,,66,Parasystole
G,Ventricular tachyarrhythmias,70,Ventricular tachycardia
,,71,"Ventricular tachycardia, unsustained"
,,72,"Ventricular tachycardia, polymorphous"
,,73,"Ventricular tachycardia, torsades de pointes"
,,74,Ventricular fibrillation
,,75,Fascicular tachycardia
,,76,Wide-QRS tachycardia
H,Atrioventricular conduction,80,Short PR interval
,,81,AV conduction ratio N:D
,,82,Prolonged PR interval
,,83,"Second-degree AV block, Mobitz type I (Wenckebach)"
,,84,"Second-degree AV block, Mobitz type II"
,,85,2:1 AV block
,,86,"AV block, varying conduction"
,,87,"AV block, advanced (high-grade)"
,,88,"AV block, complete (third-degree)"
,,89,AV dissociation
I,Intraventricular and intra-atrial conduction,100,Aberrant conduction of supraventricular beat(s)
,,101,Left anterior fascicular block
,,102,Left posterior fascicular block
,,104,Left bundle-branch block
,,105,Incomplete right bundle-branch block
,,106,Right bundle-branch block
,,107,Intraventricular conduction delay
,,108,Ventricular preexcitation
,,109,Right atrial conduction abnormality
,,110,Left atrial conduction abnormality
,,111,Epsilon wave
J,Axis and voltage,120,Right-axis deviation
,,121,Left-axis deviation
,,122,Right superior axis
,,123,Indeterminate axis
,,124,Electrical alternans
,,125,Low voltage
,,128,Abnormal precordial R-wave progression
,,131,Abnormal P-wave axis
K,Chamber hypertrophy or enlargement,140,Left atrial enlargement
,,141,Right atrial enlargement
,,142,Left ventricular hypertrophy
,,143,Right ventricular hypertrophy
,,144,Biventricular hypertrophy
L,"ST segment, T wave, and U wave",145,ST deviation
,,146,ST deviation with T-wave change
,,147,T-wave abnormality
,,148,Prolonged QT interval
,,149,Short QT interval
,,150,Prominent U waves
,,151,Inverted U waves
,,152,TU fusion
,,153,ST-T change due to ventricular ?hypertrophy
,,154,Osborn wave
,,155,Early repolarization
M,Myocardial infarction,160,Anterior MI
,,161,Inferior MI
,,162,Posterior MI
,,163,Lateral MI
,,165,Anteroseptal MI
,,166,Extensive anterior MI
,,173,MI in presence of left bundle-branch ?block
,,174,Right ventricular MI
N,Pacemaker,180,Atrial-paced complex(es) or rhythm
,,181,Ventricular-paced complex(es) or rhythm
,,182,Ventricular pacing of non–right ventricular ?apical origin
,,183,Atrial-sensed ventricular-paced ?complex(es) or rhythm
,,184,AV dual-paced complex(es) or rhythm
,,185,"Failure to capture, atrial"
,,186,"Failure to capture, ventricular"
,,187,"Failure to inhibit, atrial"
,,188,"Failure to inhibit, ventricular"
,,189,"Failure to pace, atrial"
,,190,"Failure to pace, ventricular"
"""
    ),
    dtype=str,
)

# df_primary_statements = df_primary_statements.fillna(method="ffill")
df_primary_statements = df_primary_statements.ffill(axis=0)


df_secondary_statements = pd.read_csv(
    io.StringIO(
        """
Group,Code,Description
Suggests,200,Acute pericarditis
,201,Acute pulmonary embolism
,202,Brugada abnormality
,203,Chronic pulmonary disease
,204,CNS disease
,205,Digitalis effect
,206,Digitalis toxicity
,207,Hypercalcemia
,208,Hyperkalemia
,209,Hypertrophic cardiomyopathy
,210,Hypocalcemia
,211,Hypokalemia or drug effect
,212,Hypothermia
,213,Ostium primum ASD
,214,Pericardial effusion
,215,Sinoatrial disorder
Consider,220,Acute ischemia
,221,AV nodal reentry
,222,AV reentry
,223,Genetic repolarization abnormality
,224,High precordial lead placement
,225,Hypothyroidism
,226,Ischemia
,227,Left ventricular aneurysm
,228,Normal variant
,229,Pulmonary disease
,230,Dextrocardia
,231,Dextroposition
"""
    ),
    dtype=str,
)

# df_secondary_statements = df_secondary_statements.fillna(method="ffill")
df_secondary_statements = df_secondary_statements.ffill(axis=0)


df_modifiers = pd.read_csv(
    io.StringIO(
        """
Category,Code,Description,
General,301,Borderline,
,303,Increased,
,304,Intermittent,
,305,Marked,
,306,Moderate,
,307,Multiple,
,308,Occasional,
,309,One,
,310,Frequent,
,312,Possible,
,313,Postoperative,
,314,Predominant,
,315,Probable,
,316,Prominent,
,317,(Specified) Lead(s),
,318,(Specified) Electrode(s),
,321,Nonspecific,
General: conjunctions,302,Consider,
,310,Or,
,320,And,
,319,With,
,322,Versus,
Myocardial infarction,330,Acute,
,331,Recent,
,332,Old,
,333,Of indeterminate age,
,334,Evolving,
Arrhythmias and tachyarrhythmias,340,Couplets,
,341,In a bigeminal pattern,
,342,In a trigeminal pattern,
,343,Monomorphic,
,344,Multifocal,
,345,Unifocal,
,346,With a rapid ventricular response,
,347,With a slow ventricular response,
,348,With capture beat(s),
,349,With aberrancy,
,350,Polymorphic,
Repolarization abnormalities,360,≥0.1 mV,
,361,≥0.2 mV,
,362,Depression,
,363,Elevation,
,364,Maximally toward lead,
,365,Maximally away from lead,
,366,Low amplitude,
,367,Inversion,
,369,Postpacing (anamnestic),
"""
    ),
    dtype=str,
)

# df_modifiers = df_modifiers.fillna(method="ffill")
df_modifiers = df_modifiers.ffill(axis=0)


df_comparison_statements = pd.read_csv(
    io.StringIO(
        """
Code,Statement,Criteria
400,No significant change,"Intervals (PR, QRS, QTc) remain normal or within 10% of a previously abnormal value"
,,No new or deleted diagnoses with the exception of normal variant diagnoses
401,Significant change in rhythm,New or deleted rhythm diagnosis
,,HR change >20 bpm and <50 or >100 bpm
,,New or deleted pacemaker diagnosis
402,New or worsened ischemia or infarction,"Added infarction, ST-ischemia, or T-wave-ischemia diagnosis, or worsened ST deviation or T-wave abnormality"
403,New conduction abnormality,Added AV or IV conduction diagnosis
404,Significant repolarization change,New or deleted QT diagnosis
,,New or deleted U-wave diagnosis
,,New or deleted nonischemic ST or T-wave diagnosis
,,Change in QTc >60 ms
405,Change in clinical status,"New or deleted diagnosis from Axis and Voltage, Chamber Hypertrophy, or Enlargement primary statement categories or “Suggests…” secondary statement category"
406,Change in interpretation without significant change in waveform,"Used when a primary or secondary statement is added or removed despite no real change in the tracing; ie, an interpretive disagreement exists between the readers of the first and second ECGs"
"""
    ),
    dtype=str,
)

# df_comparison_statements = df_comparison_statements.fillna(method="ffill")
df_comparison_statements = df_comparison_statements.ffill(axis=0)


df_convenience_statements = pd.read_csv(
    io.StringIO(
        """
Code,Statement
500,Nonspecific ST-T abnormality
501,ST elevation
502,ST depression
503,LVH with ST-T changes
"""
    ),
    dtype=str,
)


df_secondary_primary_statement_pairing_rules = pd.read_csv(
    io.StringIO(
        """
Secondary Code,May Accompany These Primary Codes
200,145-147
201,"21, 105, 109, 120, 131, 141, 145-147"
202,"105, 106, 145-146"
203,"109, 120, 125, 128, 131, 141, 143"
204,147
205,145-147
206,145-147
207,149
208,147
209,142
210,148
211,"147-148, 150"
212,"14, 154"
213,"82, 105-106, 121"
214,124
215,"42, 131, 145-147"
220,"145-147, 151"
221,"55, 56"
222,"55, 56"
223,"148, 149"
224,128
225,"22, 24-26, 37, 38"
226,145-147
227,145-147
228,"80, 105, 128, 155"
229,"109, 120, 122-123, 125, 128, 131, 141, 143"
230,"128, 131"
231,128
"""
    ),
    dtype=str,
)


df_general_modifier_primary_statement_pairing_rules = pd.read_csv(
    io.StringIO(
        """
General Modifier Code,May (May Not) Accompany These Primary Codes or May Be Between Codes in These Categories or Groups of Categories,May/May Not,Location
301,"1-20, 24-76, 81, 83-106, 108, 122-124",May not,b
302,"1-3, 12-16, 80-82, 111-130, 145-152",May not,"b, i"
303,"30, 31, 36, 37, 41, 60, 62, 63, 82, 107, 109, 110",May,"a, b"
304,"21-26, 30-76, 80, 82-108, 124, 180-190",May,b
305,"1-20, 27-76, 81, 85-106, 111, 122, 123, 148-150, 160-190",May not,b
306,"1-20, 27-76, 81, 85-106, 111, 122, 123, 148-150, 160-190",May not,b
307,"26, 30, 31, 36, 37, 41, 60-62, 185-190",May,b
308,"26, 30, 31, 36, 37, 41, 60-62, 185-190",May,b
309,"26, 30, 31, 36, 37, 41, 60-62, 185-190",May,b
310,"C, D, E, F, G, N, H, I, J, K, L, M",May,i
312,"1-3, 15, 80-82, 120-122, 128",May not,b
313,145-147,May,b
314,"20-23, 33-35, 38-56, 63-76, 83-89, 180-184",May,b
315,"1-3, 15, 80-82, 120-122, 128",May not,b
316,"1-20, 27-76, 81, 85-106, 111, 122, 123, 148-150, 160-190",May not,b
317,"C, D, E, F, G, N, H, I, J, K, L, M",May,i
318,"C, D, E, F, G, N, H, I, J, K, L, M",May,i
319,"C, D, E, F, G, N, 100, J, K, L, M",May,i
321,"40, 55, 56, 145-147",May,b
"""
    ),
    dtype=str,
)

df_general_modifier_primary_statement_pairing_rules["Location"] = df_general_modifier_primary_statement_pairing_rules[
    "Location"
].apply(lambda location: location.replace("b", "before").replace("a", "after").replace("i", "between"))
