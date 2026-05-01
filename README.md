# Final-Project
AI 100 Final Project — Heart Disease Classification
Overview
This project intentionally introduces 10 bugs into a tabular ML system (Random Forest on synthetic heart disease data) and uses GenAI (Claude) as a Socratic cognitive partner to evaluate debugging reflections.
Files
FileDescriptionmodel.pyBase ML system (working)bugs.pyAll 10 buggy variants with captured error outputbug_cases.xlsxRequired Google Sheet / Excel with all 10 casesreport.pdfPDF report on GenAI and AI system building lessonsevaluations.jsonRaw GenAI evaluation responses
System Description

Task: Binary classification — predict heart disease presence (0/1)
Dataset: Synthetic tabular data (500 samples, 13 features) modeled on the Cleveland Heart Disease dataset
Model: sklearn.ensemble.RandomForestClassifier (100 estimators, max_depth=5, random_state=42)
Baseline Accuracy: 87.0%

Bug Summary
#ChangeError TypeGenAI Label1test_size=1.5InvalidParameterErrorBad2fit_transform on test setSilent (data leakage)Bad3y instead of y_train in fitValueErrorGood4n_estimators="100" (string)InvalidParameterErrorGood5predict before fitNotFittedErrorGood6Wrong feature count at predictValueErrorGood7No random_stateSilent (non-reproducible)Bad8stratify=y_train (wrong length)ValueErrorGood9max_depth=0InvalidParameterErrorGood10Swapped metric argsSilent (wrong semantics)Bad
