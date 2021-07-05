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
from easydict import EasyDict as ED


__all__ = [
    "AF", "AFL",  # atrial
    "IAVB", "LBBB", "RBBB", "CRBBB", "IRBBB", "LAnFB", "NSIVCB",  # conduction block
    "PAC", "PJC", "PVC", "SPB",  # premature: qrs, morphology
    "LPR", "LQT", "QAb", "TAb", "TInv",  # wave morphology
    "LAD", "RAD",  # axis
    "Brady", "LQRSV",  # qrs (RR interval, amplitude)
    "SA", "SB", "NSR", "STach",  # sinus
    "PR",  # pacer
    "STD", "STE",  # ST segments
]


AF = ED({  # rr, morphology
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
})

AFL = ED({  # rr, morphology
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
})

Brady = ED({  # rr
    "fullname": "bradycardia",
    "url": [
        "https://litfl.com/bradycardia-ddx/",
        "https://en.wikipedia.org/wiki/Bradycardia"
    ],
    "knowledge": [
        "heart rate (ventricular rate) < 60/min in an adult",
    ],
})

IAVB = {  # morphology
    "fullname": "1st degree av block",
    "url": [
        "https://litfl.com/first-degree-heart-block-ecg-library/",
        "https://en.wikipedia.org/wiki/Atrioventricular_block#First-degree_Atrioventricular_Block"
    ],
    "knowledge": [
        "PR interval > 200ms",
        "Marked’ first degree block if PR interval > 300ms",
        "P waves might be buried in the preceding T wave",
        "there are no dropped, or skipped, beats",
    ],
}

LBBB = ED({  # morphology
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
})

RBBB = ED({  # morphology
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
})

CRBBB = ED({  # morphology
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
})

IRBBB = ED({  # morphology
    "fullname": "incomplete right bundle branch block",
    "url": [
        "https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/",
        "https://en.wikipedia.org/wiki/Right_bundle_branch_block#Diagnosis",
    ],
    "knowledge": [
        "defined as an RSR’ pattern in V1-3 with QRS duration < 120ms (and > 100ms?)",
        "normal variant, commonly seen in children (of no clinical significance)",
    ],
})

LAnFB = ED({  # morphology
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
        "LAD of degree (-45°, -90°)"
    ],
})

LQRSV = ED({  # voltage
    "fullname": "low qrs voltages",
    "url": [
        "https://litfl.com/low-qrs-voltage-ecg-library/",
    ],
    "knowledge": [
        "amplitudes of all the QRS complexes in the limb leads are < 5mm (0.5mV); or  amplitudes of all the QRS complexes in the precordial leads are < 10mm (1mV)",
    ],
})

NSIVCB = ED({  # mophology
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
})

PR = ED({  # morphology
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
})

PAC = ED({  # morphology, very complicated
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
})

PJC = ED({  # morphology
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
})

PVC = ED({  # morphology
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
})

LPR = ED({  # morphology
    "fullname": "prolonged pr interval",
    "url": [
        "https://litfl.com/pr-interval-ecg-library/",
        "https://en.wikipedia.org/wiki/PR_interval",
        "https://www.healio.com/cardiology/learn-the-heart/ecg-review/ecg-interpretation-tutorial/pr-interval",
    ],
    "knowledge": [
        "PR interval >200ms",
    ],
})

LQT = ED({  # morphology
    "fullname": "prolonged qt interval",
    "url": [
        "https://litfl.com/qt-interval-ecg-library/",
        "https://en.wikipedia.org/wiki/Long_QT_syndrome",
    ],
    "knowledge": [
        "LQT is measured by QTc (see ref url)"
        "QTc is prolonged if > 440ms in men or > 460ms in women (or > 480ms?)",
    ],
})

QAb = ED({   # morphology
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
})

RAD = ED({  # morphology
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
})

LAD = ED({  # morphology
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
})

SA = ED({  # morphology
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
})

SB = ED({  # rr, morphology
    "fullname": "sinus bradycardia",
    "url": [
        "https://litfl.com/sinus-bradycardia-ecg-library/",
        "https://en.wikipedia.org/wiki/Sinus_bradycardia",
    ],
    "knowledge": [
        "sinus rhythm (NSR); with a resting heart rate of < 60 bpm in adults (Brady)",
    ],
})

NSR = ED({  # rr, morphology
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
})

STach = ED({  # rr, morphology,
    "fullname": "sinus tachycardia",
    "url": [
        "https://litfl.com/sinus-tachycardia-ecg-library/",
        "https://litfl.com/tachycardia-ddx/",
        "https://en.wikipedia.org/wiki/Sinus_tachycardia",
    ],
    "knowledge": [
        "sinus rhythm (NSR), with a resting heart rate of > 100 bpm in adults",
    ],
})

SVPB = ED({  # morphology, equiv. to PAC for CINC2020
    "fullname": "supraventricular premature beats",
    "url": [
        "https://en.wikipedia.org/wiki/Premature_atrial_contraction#Supraventricular_extrasystole",
    ],
    "knowledge": PAC["knowledge"] + PJC["knowledge"],
})

TAb = ED({  # morphology
    "fullname": "t wave abnormal",
    "url": [
        "https://litfl.com/t-wave-ecg-library/",
        "https://en.wikipedia.org/wiki/T_wave",
    ],
    "knowledge": [
        "normal T wave: upright in all leads, except aVR, aVL, III and V1; amplitude < 5mm in limb leads, < 10mm in precordial leads; asymmetrical with a rounded peak",
        "abnormalities: peaked (amplitude) T waves; hyperacute T waves (broad, symmetrical, usually with increased amplitude); inverted T waves (TInv); biphasic T waves; ‘camel hump’ T waves; flattened T waves (± 0.1mV)"
    ],
})

TInv = ED({  # morphology
    "fullname": "t wave inversion",
    "url": [
        "https://en.wikipedia.org/wiki/T_wave#Inverted_T_wave",
        "https://litfl.com/t-wave-ecg-library/",
    ],
    "knowledge": [
        "normal T wave should be upright (positive peak amplitude) in all leads, except aVR, aVL, III and V1",
    ],
})

VPB = ED({
    "fullname": "ventricular premature beats",
    "url": PVC["url"],
    "knowledge": PVC["knowledge"],
})

SPB = SVPB  # alias

STD = ED({
    "fullname": "st depression",
    "url": [
        "https://litfl.com/st-segment-ecg-library/",
        "https://ecgwaves.com/st-segment-normal-abnormal-depression-elevation-causes/",
        "https://en.wikipedia.org/wiki/ST_elevation",
    ],
    "knowledge": [
        "",
    ],
})

STE = ED({
    "fullname": "st elevation",
    "url": [
        "https://litfl.com/st-segment-ecg-library/",
        "https://ecgwaves.com/st-segment-normal-abnormal-depression-elevation-causes/",
        "https://en.wikipedia.org/wiki/ST_depression",
    ],
    "knowledge": [
        "",
    ],
})
