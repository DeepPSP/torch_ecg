# -*- coding: utf-8 -*-
"""
knowledge about ECG arrhythmia

Standard 12Leads ECG
--------------------
    Inferior leads: II, III, aVF
    Lateral leads: I, aVL, V5-6
    Septal leads: aVR, V1
    Anterior leads: V2-4
    -----------------------------------
    Chest (precordial) leads: V1-6
    Limb leads: I, II, III, aVR, aVL, aVF

ECG rhythm (https://litfl.com/ecg-rhythm-evaluation/)
-----------------------------------------------------
1. On a 12 lead ECG, ECG rhythm is usually a 10 second recording from Lead II
2. 7 steps to analyze:
    2.1. rate (brady < 60 bpm; 60 bpm <= normal <= 100 bpm; tachy > 100 bpm)
    2.2. pattern (temporal) of QRS complex (regular or irregular; if irregular, regularly irregular or irregularly irregular)
    2.3. morphology (spatial) of QRS complex (narrow <= 120 ms; wide > 120 ms)
    2.4. P waves (absent or present; morphology; PR interval)
    2.5. relation of P waves and QRS complexes (atrial rate and ventricular rate, AV association or AV disassociation)
    2.6. onset and termination (abrupt or gradual)
    (2.7. response to vagal manoeuvres)

ECG waves
---------
https://ecgwaves.com/topic/ecg-normal-p-wave-qrs-complex-st-segment-t-wave-j-point/

References
----------
[1] https://litfl.com/
[2] https://ecgwaves.com/
[3] https://ecglibrary.com/ecghome.php
[4] https://courses.cs.washington.edu/courses/cse466/13au/pdfs/lectures/ECG%20filtering.pdf

NOTE that wikipedia is NOT listed in the References

"""
from io import StringIO

import pandas as pd

from ..cfg import CFG

__all__ = [
    # named lead sets
    "Standard12Leads",
    "ChestLeads",
    "PrecordialLeads",
    "LimbLeads",
    "InferiorLeads",
    "LateralLeads",
    "SeptalLeads",
    "AnteriorLeads",
    # ECG abnormalities (arrhythmias)
    "AF",
    "AFL",  # atrial
    "IAVB",
    "LBBB",
    "CLBBB",
    "RBBB",
    "CRBBB",
    "IRBBB",
    "LAnFB",
    "NSIVCB",
    "BBB",  # conduction block
    "PAC",
    "PJC",
    "PVC",
    "SPB",  # premature: qrs, morphology
    "LPR",
    "LQT",
    "QAb",
    "TAb",
    "TInv",  # wave morphology
    "LAD",
    "RAD",  # axis
    "Brady",
    "LQRSV",
    "PRWP",  # qrs (RR interval, amplitude)
    "SA",
    "SB",
    "NSR",
    "STach",  # sinus
    "PR",  # pacer
    "STD",
    "STE",  # ST segments
]


Standard12Leads = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
ChestLeads = [f"V{n}" for n in range(1, 7)]
PrecordialLeads = ChestLeads
LimbLeads = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
]
InferiorLeads = [
    "II",
    "III",
    "aVF",
]
LateralLeads = [
    "I",
    "aVL",
    "V5",
    "V6",
]
SeptalLeads = [
    "aVR",
    "V1",
]
AnteriorLeads = [
    "V2",
    "V3",
    "V4",
]


_snomedbrowser_url = "https://snomedbrowser.com/Codes/Details/"


AF = CFG(
    {  # rr, morphology
        "fullname": "atrial fibrillation",
        "url": [
            "https://litfl.com/atrial-fibrillation-ecg-library/",
            "https://en.wikipedia.org/wiki/Atrial_fibrillation#Screening",
        ],
        "knowledge": [
            "irregularly irregular rhythm",
            "no P waves",
            "absence of an isoelectric baseline",
            "variable ventricular rate",
            "QRS complexes usually < 120 ms unless pre-existing bundle branch block, accessory pathway, or rate related aberrant conduction",
            "fibrillatory waves (f-wave) may be present and can be either fine (amplitude < 0.5mm) or coarse (amplitude > 0.5mm)",
            "fibrillatory waves (f-wave) may mimic P waves leading to misdiagnosis",
        ],
    }
)

AFL = CFG(
    {  # rr, morphology
        "fullname": "atrial flutter",
        "url": [
            "https://litfl.com/atrial-flutter-ecg-library/",
            "https://en.wikipedia.org/wiki/Atrial_flutter",
        ],
        "knowledge": [
            "a type of supraventricular tachycardia caused by a re-entry circuit within the right atrium",
            "fairly predictable atrial rate (NOT equal to ventricular rate for AFL) of around 300 bpm (range 200-400)",
            "fixed AV blocks, with ventricular rate a fraction (1/2,1/3,etc.) of atrial rate",
            "narrow complex tachycardia (ref. supraventricular & ventricular rate)",
            "flutter waves ('saw-tooth' pattern) best seen in leads II, III, aVF (may be more easily spotted by turning the ECG upside down), may resemble P waves in V1",
            "loss of the isoelectric baseline",  # important
        ],
    }
)

Brady = CFG(
    {  # rr
        "fullname": "bradycardia",
        "url": [
            "https://litfl.com/bradycardia-ddx/",
            "https://en.wikipedia.org/wiki/Bradycardia",
        ],
        "knowledge": [
            "heart rate (ventricular rate) < 60/min in an adult",
        ],
    }
)

BBB = CFG(
    {  # morphology
        "fullname": "bundle branch block",
        "url": [],
        "knowledge": [],
    }
)

IAVB = CFG(
    {  # morphology
        "fullname": "1st degree av block",
        "url": [
            "https://litfl.com/first-degree-heart-block-ecg-library/",
            "https://en.wikipedia.org/wiki/Atrioventricular_block#First-degree_Atrioventricular_Block",
        ],
        "knowledge": [
            "PR interval > 200ms",
            "Marked’ first degree block if PR interval > 300ms",
            "P waves might be buried in the preceding T wave",
            "there are no dropped, or skipped, beats",
        ],
    }
)

LBBB = CFG(
    {  # morphology
        "fullname": "left bundle branch block",
        "url": [
            "https://litfl.com/left-bundle-branch-block-lbbb-ecg-library/",
            "https://en.wikipedia.org/wiki/Left_bundle_branch_block",
        ],
        "knowledge": [
            "heart rhythm must be supraventricular",
            "QRS duration of > 120 ms",
            "lead V1: Dominant S wave, with QS or rS complex",
            "lateral leads: M-shaped, or notched, or broad monophasic R wave or RS complex; absence of Q waves (small Q waves are still allowed in aVL)",
            "chest (precordial) leads: poor R wave progression",
            "left precordial leads (V5-6): prolonged R wave peak time > 60ms",
            "ST segments and T waves always go in the opposite direction to the main vector of the QRS complex",
        ],
    }
)

CLBBB = CFG(
    {  # morphology
        "fullname": "complete left bundle branch block",
        "url": [],
        "knowledge": [],
    }
)

RBBB = CFG(
    {  # morphology
        "fullname": "right bundle branch block",
        "url": [
            "https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/",
            "https://en.wikipedia.org/wiki/Right_bundle_branch_block",
        ],
        "knowledge": [
            "broad QRS > 100 ms (incomplete block) or > 120 ms (complete block)",
            "leads V1-3: RSR’ pattern (‘M-shaped’ QRS complex); sometimes a broad monophasic R wave or a qR complex in V1",
            "lateral leads: wide, slurred S wave",
        ],
    }
)

CRBBB = CFG(
    {  # morphology
        "fullname": "complete right bundle branch block",
        "url": [
            "https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/",
            "https://en.wikipedia.org/wiki/Right_bundle_branch_block",
        ],
        "knowledge": [
            "broad QRS > 120 ms",
            "leads V1-3: RSR’ pattern (‘M-shaped’ QRS complex); sometimes a broad monophasic R wave or a qR complex in V1",
            "lateral leads: wide, slurred S wave",
        ],
    }
)

IRBBB = CFG(
    {  # morphology
        "fullname": "incomplete right bundle branch block",
        "url": [
            "https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/",
            "https://en.wikipedia.org/wiki/Right_bundle_branch_block#Diagnosis",
        ],
        "knowledge": [
            "defined as an RSR’ pattern in V1-3 with QRS duration < 120ms (and > 100ms?)",
            "normal variant, commonly seen in children (of no clinical significance)",
        ],
    }
)

LAnFB = CFG(
    {  # morphology
        "fullname": "left anterior fascicular block",
        "url": [
            "https://litfl.com/left-anterior-fascicular-block-lafb-ecg-library/",
            "https://en.wikipedia.org/wiki/Left_anterior_fascicular_block",
        ],
        "knowledge": [
            "inferior leads (II, III, aVF): small R waves, large negative voltages (deep S waves), i.e. 'rS complexes'",
            "left-sided leads (I, aVL): small Q waves, large positive voltages (tall R waves), i.e. 'qR complexes'",
            "slight widening of the QRS",
            "increased R wave peak time in aVL",
            "LAD of degree (-45°, -90°)",
        ],
    }
)

LQRSV = CFG(
    {  # voltage
        "fullname": "low qrs voltages",
        "url": [
            "https://litfl.com/low-qrs-voltage-ecg-library/",
            "https://www.healio.com/cardiology/learn-the-heart/ecg-review/ecg-topic-reviews-and-criteria/low-voltage-review",
        ],
        "knowledge": [
            "peak-to-peak (VERY IMPORTANT) amplitudes of all the QRS complexes in the limb leads are < 5mm (0.5mV); or amplitudes of all the QRS complexes in the precordial leads are < 10mm (1mV)",
        ],
    }
)

PRWP = CFG(
    {  # voltage
        "fullname": "poor R wave progression",
        "url": [
            "https://litfl.com/poor-r-wave-progression-prwp-ecg-library/",
            "https://www.healio.com/cardiology/learn-the-heart/ecg-review/ecg-topic-reviews-and-criteria/poor-r-wave-progression",
            "https://emergencymedicinecases.com/ecg-cases-poor-r-wave-progression-late-mnemonic/",
            "https://www.wikidoc.org/index.php/Poor_R_Wave_Progression",
        ],
        "knowledge": [  # The definition of poor R wave progression (PRWP) varies in the literature
            "absence of the normal increase in size of the R wave in the precordial leads when advancing from lead V1 to V6",
            "in lead V1, the R wave should be small. The R wave becomes larger throughout the precordial leads, to the point where the R wave is larger than the S wave in lead V4. The S wave then becomes quite small in lead V6.",
            "failure of the R wave to progress in amplitude (R<3mm in V3), reversal of the progression (eg R in V2>V3), or delayed transition beyond V4",
            "R wave is less than 2–4 mm in leads V3 or V4 and/or there is presence of a reversed R wave progression, which is defined as R in V4 < R in V3 or R in V3 < R in V2 or R in V2 < R in V1, or any combination of these",
        ],
    }
)

NSIVCB = CFG(
    {  # mophology
        "fullname": "nonspecific intraventricular conduction disorder",
        "url": [
            "https://ecgwaves.com/topic/nonspecific-intraventricular-conduction-delay-defect/",
            "https://www.dynamed.com/condition/intraventricular-conduction-disorders-including-left-and-right-bundle-branch-block-lbbb-and-rbbb",
            "https://www.sciencedirect.com/science/article/pii/S0735109708041351",
            "https://www.heartrhythmjournal.com/article/S1547-5271(15)00073-9/abstract",
        ],
        "knowledge": [
            "widended (> 110ms) QRS complex, with not meeting the criteria (morphology different from) for LBBB and RBBB",
        ],
    }
)

PR = CFG(
    {  # morphology
        "fullname": "pacing rhythm",
        "url": [
            "https://litfl.com/pacemaker-rhythms-normal-patterns/",
            "https://www.youtube.com/watch?v=YkB4oX_COi8",
        ],
        "knowledge": [
            "morphology is dependent on the pacing mode (AAI, VVI, DDD, Magnet) used",
            "there are pacing spikes: vertical spikes of short duration, usually 2 ms (in doubt, viewing signals of the CINC2020 dataset, probably 10-20ms(at most half a small grid)?)",  # important
            "AAI (atrial pacing): pacing spike precedes the p wave",
            "VVI (ventricle pacing): pacing spike precedes the QRS complex; morphology similar to LBBB or RBBB (depending on lead placement)",
        ],
    }
)

PAC = CFG(
    {  # morphology, very complicated
        "fullname": "premature atrial contraction",
        "url": [
            "https://litfl.com/premature-atrial-complex-pac/",
            "https://en.wikipedia.org/wiki/Premature_atrial_contraction",
        ],
        "knowledge": [
            "an abnormal (non-sinus) P wave is followed by a QRS complex",
            "P wave typically has a different morphology and axis to the sinus P waves",
            "abnormal P wave may be hidden in the preceding T wave, producing a “peaked” or “camel hump” appearance",
            # to add more
        ],
    }
)

PJC = CFG(
    {  # morphology
        "fullname": "premature junctional contraction",
        "url": [
            "https://litfl.com/premature-junctional-complex-pjc/",
            "https://en.wikipedia.org/wiki/Premature_junctional_contraction",
        ],
        "knowledge": [
            "narrow QRS complex, either (1) without a preceding P wave or (2) with a retrograde P wave which may appear before, during, or after the QRS complex. If before, there is a short PR interval of < 120 ms and the  “retrograde” P waves are usually inverted in leads II, III and aVF",
            "occurs sooner than would be expected for the next sinus impulse",
            "followed by a compensatory pause",
        ],
    }
)

PVC = CFG(
    {  # morphology
        "fullname": "premature ventricular contractions",
        "url": [
            "https://litfl.com/premature-ventricular-complex-pvc-ecg-library/",
            "https://en.wikipedia.org/wiki/Premature_ventricular_contraction",
        ],
        "knowledge": [
            "broad QRS complex (≥ 120 ms) with abnormal morphology",
            "premature — i.e. occurs earlier than would be expected for the next sinus impulse",
            "discordant ST segment and T wave changes",
            "usually followed by a full compensatory pause",
            "retrograde capture of the atria may or may not occur",
        ],
    }
)

LPR = CFG(
    {  # morphology
        "fullname": "prolonged pr interval",
        "url": [
            "https://litfl.com/pr-interval-ecg-library/",
            "https://en.wikipedia.org/wiki/PR_interval",
            "https://www.healio.com/cardiology/learn-the-heart/ecg-review/ecg-interpretation-tutorial/pr-interval",
        ],
        "knowledge": [
            "PR interval >200ms",
        ],
    }
)

LQT = CFG(
    {  # morphology
        "fullname": "prolonged qt interval",
        "url": [
            "https://litfl.com/qt-interval-ecg-library/",
            "https://en.wikipedia.org/wiki/Long_QT_syndrome",
        ],
        "knowledge": [
            "LQT is measured by QTc (see ref url)"
            "QTc is prolonged if > 440ms in men or > 460ms in women (or > 480ms?)",
        ],
    }
)

QAb = CFG(
    {  # morphology
        "fullname": "qwave abnormal",
        "url": [
            "https://litfl.com/q-wave-ecg-library/",
            "https://en.ecgpedia.org/wiki/Pathologic_Q_Waves",
            "https://wikem.org/wiki/Pathologic_Q_waves",
        ],
        "knowledge": [
            "> 40 ms (1 mm) wide; > 2 mm deep; > 1/4 of depth of QRS complex in ANY TWO leads of a contiguous lead grouping: I, aVL,V6; V4–V6; II, III, aVF",
            "seen (≥ 0.02 s or QS complex) in leads V1-3",
        ],
    }
)

RAD = CFG(
    {  # morphology
        "fullname": "right axis deviation",
        "url": [
            "https://litfl.com/right-axis-deviation-rad-ecg-library/",
            "https://en.wikipedia.org/wiki/Right_axis_deviation",
            "https://www.ncbi.nlm.nih.gov/books/NBK470532/",
            "https://ecglibrary.com/axis.html",
            "https://www.youtube.com/watch?v=PbzDN2_rAFc",
        ],
        "knowledge": [  # should combine with LAD to study
            "QRS axis greater than +90°",
            "2-lead method: lead I is NEGATIVE; lead aVF is POSITIVE",  # important, ref LAD
            "3-lead method: lead I is NEGATIVE; lead aVF is POSITIVE (and II, III)",  # important, ref LAD
        ],
    }
)

LAD = CFG(
    {  # morphology
        "fullname": "left axis deviation",
        "url": [
            "https://litfl.com/left-axis-deviation-lad-ecg-library/",
            "https://en.wikipedia.org/wiki/Left_axis_deviation",
            "https://www.ncbi.nlm.nih.gov/books/NBK470532/",
            "https://ecglibrary.com/axis.html",
            "https://www.youtube.com/watch?v=PbzDN2_rAFc",
        ],
        "knowledge": [  # should combine with RAD to study
            "QRS axis (-30°, -90°)",
            "2-lead method: lead I is POSITIVE; lead aVF is NEGATIVE",  # important, ref RAD
            "3-lead method: POSITIVE in leads I (and aVL?); NEGATIVE in leads II, aVF, (and III?)",  # important, ref RAD
            "LAnFB, LBBB, PR, ventricular ectopics are causes of LAD",
        ],
    }
)

SA = CFG(
    {  # morphology
        "fullname": "sinus arrhythmia",
        "url": [
            "https://litfl.com/sinus-arrhythmia-ecg-library/",
            "https://www.healthline.com/health/sinus-arrhythmia",
        ],
        "knowledge": [
            "sinus rhythm (NSR), with a beat-to-beat variation (more than 120 ms) in the PP interval, producing an irregular ventricular rate",
            "PP interval gradually lengthens and shortens in a cyclical fashion, usually corresponding to the phases of the respiratory cycle",
            "normal sinus P waves (upright in leads I and II) with a constant morphology (i.e. not PAC)",
            "constant PR interval (i.e. not Mobitz I AV block)",
        ],
    }
)

SB = CFG(
    {  # rr, morphology
        "fullname": "sinus bradycardia",
        "url": [
            "https://litfl.com/sinus-bradycardia-ecg-library/",
            "https://en.wikipedia.org/wiki/Sinus_bradycardia",
        ],
        "knowledge": [
            "sinus rhythm (NSR); with a resting heart rate of < 60 bpm in adults (Brady)",
        ],
    }
)

NSR = CFG(
    {  # rr, morphology
        "fullname": "sinus rhythm",  # the NORMAL rhythm
        "url": [
            "https://litfl.com/normal-sinus-rhythm-ecg-library/",
            "https://en.wikipedia.org/wiki/Sinus_rhythm",
        ],
        "knowledge": [
            "regular rhythm (< 0.16 s variation in the shortest and longest durations between successive P waves) at a rate of 60-100 bpm",
            "each QRS complex is preceded by a normal P wave (positive in lead I, lead II, and aVF; negative in lead aVR; any of biphasic (-/+), positive or negative in lead aVL; positive in all chest leads, except for V1 which may be biphasic (+/-))",
            "normal PR interval, QRS complex and QT interval",  # normal
        ],
    }
)

STach = CFG(
    {  # rr, morphology,
        "fullname": "sinus tachycardia",
        "url": [
            "https://litfl.com/sinus-tachycardia-ecg-library/",
            "https://litfl.com/tachycardia-ddx/",
            "https://en.wikipedia.org/wiki/Sinus_tachycardia",
        ],
        "knowledge": [
            "sinus rhythm (NSR), with a resting heart rate of > 100 bpm in adults",
        ],
    }
)

SVPB = CFG(
    {  # morphology, equiv. to PAC for CINC2020
        "fullname": "supraventricular premature beats",
        "url": [
            "https://en.wikipedia.org/wiki/Premature_atrial_contraction#Supraventricular_extrasystole",
        ],
        "knowledge": PAC["knowledge"] + PJC["knowledge"],
    }
)

TAb = CFG(
    {  # morphology
        "fullname": "t wave abnormal",
        "url": [
            "https://litfl.com/t-wave-ecg-library/",
            "https://en.wikipedia.org/wiki/T_wave",
        ],
        "knowledge": [
            "normal T wave: upright in all leads, except aVR, aVL, III and V1; amplitude < 5mm in limb leads, < 10mm in precordial leads; asymmetrical with a rounded peak",
            "abnormalities: peaked (amplitude) T waves; hyperacute T waves (broad, symmetrical, usually with increased amplitude); inverted T waves (TInv); biphasic T waves; ‘camel hump’ T waves; flattened T waves (± 0.1mV)",
        ],
    }
)

TInv = CFG(
    {  # morphology
        "fullname": "t wave inversion",
        "url": [
            "https://en.wikipedia.org/wiki/T_wave#Inverted_T_wave",
            "https://litfl.com/t-wave-ecg-library/",
        ],
        "knowledge": [
            "normal T wave should be upright (positive peak amplitude) in all leads, except aVR, aVL, III and V1",
        ],
    }
)

VPB = CFG(
    {
        "fullname": "ventricular premature beats",
        "url": PVC["url"],
        "knowledge": PVC["knowledge"],
    }
)

SPB = SVPB  # alias

STD = CFG(
    {
        "fullname": "st depression",
        "url": [
            "https://litfl.com/st-segment-ecg-library/",
            "https://ecgwaves.com/st-segment-normal-abnormal-depression-elevation-causes/",
            "https://en.wikipedia.org/wiki/ST_elevation",
        ],
        "knowledge": [
            "",
        ],
    }
)

STE = CFG(
    {
        "fullname": "st elevation",
        "url": [
            "https://litfl.com/st-segment-ecg-library/",
            "https://ecgwaves.com/st-segment-normal-abnormal-depression-elevation-causes/",
            "https://en.wikipedia.org/wiki/ST_depression",
        ],
        "knowledge": [
            "",
        ],
    }
)


_dx_mapping = pd.read_csv(
    StringIO(
        """Dx,SNOMEDCTCode,Abbreviation
atrial fibrillation,164889003,AF
atrial flutter,164890007,AFL
bundle branch block,6374002,BBB
bradycardia,426627000,Brady
complete left bundle branch block,733534002,CLBBB
complete right bundle branch block,713427006,CRBBB
1st degree av block,270492004,IAVB
incomplete right bundle branch block,713426002,IRBBB
left axis deviation,39732003,LAD
left anterior fascicular block,445118002,LAnFB
left bundle branch block,164909002,LBBB
low qrs voltages,251146004,LQRSV
nonspecific intraventricular conduction disorder,698252002,NSIVCB
sinus rhythm,426783006,NSR
premature atrial contraction,284470004,PAC
pacing rhythm,10370003,PR
poor R wave Progression,365413008,PRWP
premature ventricular contractions,427172004,PVC
prolonged pr interval,164947007,LPR
prolonged qt interval,111975006,LQT
qwave abnormal,164917005,QAb
right axis deviation,47665007,RAD
right bundle branch block,59118001,RBBB
sinus arrhythmia,427393009,SA
sinus bradycardia,426177001,SB
sinus tachycardia,427084000,STach
supraventricular premature beats,63593006,SVPB
t wave abnormal,164934002,TAb
t wave inversion,59931005,TInv
ventricular premature beats,17338001,VPB
accelerated atrial escape rhythm,233892002,AAR
abnormal QRS,164951009,abQRS
atrial escape beat,251187003,AED
accelerated idioventricular rhythm,61277005,AIVR
accelerated junctional rhythm,426664006,AJR
suspect arm ecg leads reversed,251139008,ALR
acute myocardial infarction,57054005,AMI
acute myocardial ischemia,413444003,AMIs
anterior ischemia,426434006,AnMIs
anterior myocardial infarction,54329005,AnMI
atrial bigeminy,251173003,AB
atrial fibrillation and flutter,195080001,AFAFL
atrial hypertrophy,195126007,AH
atrial pacing pattern,251268003,AP
atrial rhythm,106068003,ARH
atrial tachycardia,713422000,ATach
av block,233917008,AVB
atrioventricular dissociation,50799005,AVD
atrioventricular junctional rhythm,29320008,AVJR
atrioventricular  node reentrant tachycardia,251166008,AVNRT
atrioventricular reentrant tachycardia,233897008,AVRT
blocked premature atrial contraction,251170000,BPAC
brugada,418818005,BRU
brady tachy syndrome,74615001,BTS
chronic atrial fibrillation,426749004,CAF
countercolockwise rotation,251199005,CCR
clockwise or counterclockwise vectorcardiographic loop,61721007,CVCL/CCVCL
cardiac dysrhythmia,698247007,CD
complete heart block,27885002,CHB
congenital incomplete atrioventricular heart block,204384007,CIAHB
coronary heart disease,53741008,CHD
chronic myocardial ischemia,413844008,CMI
clockwise rotation,251198002,CR
diffuse intraventricular block,82226007,DIB
early repolarization,428417006,ERe
fusion beats,13640000,FB
fqrs wave,164942001,FQRS
heart failure,84114007,HF
heart valve disorder,368009,HVD
high t-voltage,251259000,HTV
indeterminate cardiac axis,251200008,ICA
2nd degree av block,195042002,IIAVB
mobitz type II atrioventricular block,426183003,IIAVBII
inferior ischaemia,425419005,IIs
incomplete left bundle branch block,251120003,ILBBB
inferior ST segment depression,704997005,ISTD
idioventricular rhythm,49260003,IR
junctional escape,426995002,JE
junctional premature complex,251164006,JPC
junctional tachycardia,426648003,JTach
left atrial abnormality,253352002,LAA
left atrial enlargement,67741000119109,LAE
left atrial hypertrophy,446813000,LAH
lateral ischaemia,425623009,LIs
left posterior fascicular block,445211001,LPFB
left ventricular hypertrophy,164873001,LVH
left ventricular high voltage,55827005,LVHV
left ventricular strain,370365005,LVS
myocardial infarction,164865005,MI
myocardial ischemia,164861001,MIs
mobitz type i wenckebach atrioventricular block,54016002,MoI
nonspecific st t abnormality,428750005,NSSTTA
old myocardial infarction,164867002,OldMI
paroxysmal atrial fibrillation,282825002,PAF
prolonged P wave,251205003,PPW
paroxysmal supraventricular tachycardia,67198005,PSVT
paroxysmal ventricular tachycardia,425856008,PVT
p wave change,164912004,PWC
right atrial abnormality,253339007,RAAb
r wave abnormal,164921003,RAb
right atrial hypertrophy,446358003,RAH
right atrial  high voltage,67751000119106,RAHV
rapid atrial fibrillation,314208002,RAF
right ventricular hypertrophy,89792004,RVH
sinus atrium to atrial wandering rhythm,17366009,SAAWR
sinoatrial block,65778007,SAB
sinus arrest,5609005,SARR
sinus node dysfunction,60423000,SND
shortened pr interval,49578007,SPRI
decreased qt interval,77867006,SQT
s t changes,55930002,STC
st depression,429622005,STD
st elevation,164931005,STE
st interval abnormal,164930006,STIAb
supraventricular bigeminy,251168009,SVB
supraventricular tachycardia,426761007,SVT
transient ischemic attack,266257000,TIA
tall p wave,251223006,TPW
u wave abnormal,164937009,UAb
ventricular bigeminy,11157007,VBig
ventricular ectopics,164884008,VEB
ventricular escape beat,75532003,VEsB
ventricular escape rhythm,81898007,VEsR
ventricular fibrillation,164896001,VF
ventricular flutter,111288001,VFL
ventricular hypertrophy,266249003,VH
ventricular pre excitation,195060002,VPEx
ventricular pacing pattern,251266004,VPP
paired ventricular premature complexes,251182009,VPVC
ventricular tachycardia,164895002,VTach
ventricular trigeminy,251180001,VTrig
wandering atrial pacemaker,195101003,WAP
wolff parkinson white pattern,74390002,WPW"""
    )
)


for ea_str in __all__:
    ea = eval(ea_str)
    try:
        ea["url"].insert(
            0,
            f"{_snomedbrowser_url}{_dx_mapping[_dx_mapping.Abbreviation==ea_str]['SNOMEDCTCode'].values[0]}",
        )
    except Exception:
        pass
