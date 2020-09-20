#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# OPENCLASSROOMS
# Parcours Data Scientist
# Projet 7: Implémentez un modèle de scoring
# Etudiant: Eric Wendling
# Mentor: Julien Heiduk
# Date: 20/09/2020

#################################
### Import des modules Python ###
#################################

import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

import matplotlib
import matplotlib.pyplot as plt

import os
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State # new
from dash_table import DataTable

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import joblib
from joblib import dump, load
import shutil, os
import base64
import time
import datetime

##################
### Paramètres ###
##################

colors = {'background_1': '#FFFFFF', # blanc
          'background_2': '#C0C0C0', # gris
          'text_1': '#000000', # noir
          'text_2': 'rgba(255, 51, 0, 0.8)', # rouge
          'text_3': 'rgba(0, 153, 51, 0.8)', # vert
          'text_4': '#616A6B'} # gris

path_features = './features_app/'
path_results = './results_app/'

image_filename = 'shap_force_plot.png'
image_filename_ref = 'shap_force_plot_ref.png'

heroku_deploy = 1

##########################
### Import des données ###
##########################

# Jeu de données d'entraînement avec étiquettes
feature_names = joblib.load(path_features+'feature_names.joblib')

# Jeu de données de test avec étiquettes
if heroku_deploy == 0:
    test_features_ori = joblib.load('./features/test_features.joblib')
    print('Dimensions test_features_ori',test_features_ori.shape)
test_features = joblib.load(path_features+'test_features.joblib')
print('Dimensions test_features',test_features.shape)
test_ids_ = joblib.load(path_features+'test_ids.joblib')
test_pred = joblib.load(path_features+'test_pred.joblib')
test_labels = joblib.load(path_features+'test_labels.joblib')

# Jeu de données de test sans étiquette
if heroku_deploy == 0:
    test_sub_features_ori = joblib.load('./features/test_sub_features.joblib')
    print('Dimensions test_sub_features_ori',test_sub_features_ori.shape)
test_sub_features = joblib.load(path_features+'test_sub_features.joblib')
print('Dimensions test_sub_features',test_sub_features.shape)
test_sub_ids_ = joblib.load(path_features+'test_sub_ids.joblib')
test_sub_pred = joblib.load(path_features+'test_sub_pred.joblib')

# Importance des variables
feat_imp = joblib.load(path_results+'feat_imp.joblib')

# Dossiers de crédits similaires
nn_credit_test = joblib.load(path_results+'nn_credit_test.joblib')
nn_credit_test_sub = joblib.load(path_results+'nn_credit_test_sub.joblib')

##############################
### Traitement des données ###
##############################

# Conversion de la variable 'SK_ID_CURR' au format texte
# test
df_test_ids = pd.DataFrame(test_ids_)
val_str_list = []
for i in df_test_ids.index:
    val_int = df_test_ids['SK_ID_CURR'].loc[i]
    val_str = str(val_int)
    val_str = val_str+'_'
    val_str_list.append(val_str)
df_test_ids['SK_ID_CURR_str'] = val_str_list
df_test_ids.columns = ['SK_ID_CURR_num','SK_ID_CURR']
df_test_ids_ = df_test_ids[['SK_ID_CURR_num','SK_ID_CURR']]
df_test_ids = df_test_ids[['SK_ID_CURR']]

# test_sub
df_test_sub_ids = pd.DataFrame(test_sub_ids_)
val_str_list = []
for i in df_test_sub_ids.index:
    val_int = df_test_sub_ids['SK_ID_CURR'].loc[i]
    val_str = str(val_int)
    val_str = val_str+'_'
    val_str_list.append(val_str)    
df_test_sub_ids['SK_ID_CURR_str'] = val_str_list
df_test_sub_ids.columns = ['SK_ID_CURR_num','SK_ID_CURR']
df_test_sub_ids_ = df_test_sub_ids[['SK_ID_CURR_num','SK_ID_CURR']]
df_test_sub_ids = df_test_sub_ids[['SK_ID_CURR']]

# Ajout des identifiants 'SK_ID_CURR' aux jeux de test
test_features_with_id = df_test_ids.merge(test_features, left_index=True, right_index=True )
test_features_with_id = round(test_features_with_id,3)
test_sub_features_with_id = df_test_sub_ids.merge(test_sub_features, left_index=True, right_index=True )
test_sub_features_with_id = round(test_sub_features_with_id,3)

if heroku_deploy == 0:
    test_features_ori_with_id = df_test_ids.merge(test_features_ori, left_index=True, right_index=True )
    test_features_ori_with_id = round(test_features_ori_with_id,3)
    test_sub_features_ori_with_id = df_test_sub_ids.merge(test_sub_features_ori, left_index=True, right_index=True )
    test_sub_features_ori_with_id = round(test_sub_features_ori_with_id,3)

#####################
### Métriques ROC ###
#####################

# Calcul des mesures ROC sur la base des prédictions/valeurs réelles
# fpr = spécificité
# tpr = sensibilité
# thr = seuil de classification
[fpr, tpr, thr] = metrics.roc_curve(test_labels, test_pred)
df_fpr = pd.DataFrame(fpr)
df_tpr = pd.DataFrame(tpr)

# Dataframe contenant les mesures utiles à la création de la courbe ROC
df_roc = df_fpr.merge(df_tpr, left_index=True, right_index=True)
df_roc.columns = ['fpr','tpr']

# Calcul de la valeur AUC (mesure de l'aire sous la courbe ROC)
auc_val = round(metrics.auc(fpr, tpr),3)

print('AUC =',auc_val)

### Identification du seuil de classification optimal ###

# Option 1:

# On charge la variable solvability_threshold_g_norm qui correspond
# au seuil de classification optimal carractérisant le score sur la base des prédictions probas
solvability_threshold_g_norm = joblib.load(path_results+'thr_g_norm.joblib')
print('Seuil de classification optimal (opt 1) =',solvability_threshold_g_norm)

# On récupère l'indice relatif au seuil
# Cet indice nous permettra d'identifier les autres mesures en rapport
idx = np.max(np.where(thr >= solvability_threshold_g_norm))

# On charge le dataframe qui contient les valeurs du gain normalisé pour différentes valeurs du seuil
df_g_norm = joblib.load(path_results+'df_g_norm.joblib')
df_g_norm = df_g_norm.sort_values('seuil')

### Identification du gain optimal ###

# Option 1:

# Identification, dans le dataframe df_g_norm, de la valeur de seuil la plus proche du seuil optimal 
# (solvability_threshold_g_norm) dans l'objectif d'identifier le gain optimal
threshold_fit_list = []

# Pour chaque valeur du seuil
for i in df_g_norm['seuil']:
    # On calcule la différence entre les valeurs de seuils
    threshold_fit = solvability_threshold_g_norm - i
    # On ajoute la différence à la liste
    threshold_fit_list.append(abs(threshold_fit))

# On identifie la différence la plus petite (en valeur absolue)
best_threshold_fit = min(threshold_fit_list)
# On identifie le gain correspondant
id_ = threshold_fit_list.index(best_threshold_fit)
df_g_norm = df_g_norm.sort_values('g_norm',ascending=False)
g_norm_opt = df_g_norm['g_norm'].loc[id_]
print('Gain optimal (opt 1) =',g_norm_opt)

# Option 2:

# Seuil de classification optimal
solvability_threshold_g_norm = df_g_norm.iloc[0:1,10].values.tolist()[0]
solvability_threshold_g_norm = round(solvability_threshold_g_norm,4)
print('Seuil de classification optimal (opt 2) =',solvability_threshold_g_norm)

# Gain optimal
df_g_norm = df_g_norm.sort_values('g_norm',ascending=False)
g_norm_opt = df_g_norm.iloc[0:1,9].values.tolist()[0]
print('Gain optimal (opt 2) =',g_norm_opt)

df_g_norm = df_g_norm.sort_values('seuil',ascending=True)
# On limite les données de df_g_norm en vue de la présentation de
# la courbe d'évolution du gain
# df_g_norm = df_g_norm[df_g_norm['seuil'] < 0.4]

###############
### Scoring ###
###############

# Scoring test
test_pred_bin = (test_pred > solvability_threshold_g_norm)
test_pred_bin = np.array(test_pred_bin > 0) * 1
test_pred_bin = pd.DataFrame(test_pred_bin)

scoring_test = df_test_ids.reset_index(drop=True).merge(pd.DataFrame(test_pred), left_index=True, right_index=True)
scoring_test = scoring_test.merge(test_pred_bin, left_index=True, right_index=True)
scoring_test.columns = ['SK_ID_CURR','_P','_Pred']

scoring_test['_N'] = 1 - scoring_test['_P']
scoring_test = scoring_test[['SK_ID_CURR','_N','_P','_Pred']]
# On rajoute les étiquettes
scoring_test = scoring_test.merge(pd.DataFrame(test_labels).reset_index(drop=True), left_index=True, right_index=True)
scoring_test.columns = ['SK_ID_CURR','_N','_P','_Pred','_True']

# Scoring test - Mise à jour de la liste de valeurs de l'identifiant 'SK_ID_CURR' (dropdownlist)
scoring_test = scoring_test.sort_values('_P')
# La ligne de code suivante permet de filtrer la liste
list_test_credit_ids = scoring_test[(scoring_test['_P'] >= 0) & (scoring_test['_P'] <= 1)]
list_test_credit_ids = list_test_credit_ids.merge(df_test_ids_, left_on = 'SK_ID_CURR', right_on = 'SK_ID_CURR', how='left')
list_test_credit_ids = list_test_credit_ids['SK_ID_CURR_num']

scoring_test = round(scoring_test,5)

# On définit un individu (dossier de crédit) par défaut (pour l'affichage initial)
scor_test = pd.DataFrame(test_ids_).reset_index().merge(pd.DataFrame(test_pred),
                                                        left_index=True, right_index=True)[['SK_ID_CURR',0]]
scor_test = scor_test.sort_values(0)

credit_id_default_test = (scor_test[scor_test[0] < solvability_threshold_g_norm]).tail()
credit_id_default_test = (scor_test[scor_test[0] < solvability_threshold_g_norm]).tail(1).iloc[0,0]

scor_test_sub = pd.DataFrame(test_sub_ids_).reset_index().merge(pd.DataFrame(test_sub_pred), left_index=True, right_index=True)[['SK_ID_CURR',0]]
scor_test_sub = scor_test_sub.sort_values(0)
credit_id_default_test_sub = scor_test_sub[scor_test_sub[0] < solvability_threshold_g_norm].tail(1).iloc[0,0]

# Valeur des variables pour un individu (demande/dossier de crédit)
# Initialisation de la requête
feat_val_per_cust = test_features_with_id[test_features_with_id['SK_ID_CURR'] == str(credit_id_default_test)+'_'].T
feat_val_per_cust.columns = ['value']
feat_val_per_cust.reset_index(inplace=True)

# On fusionne les variables importantes avec les variables du dossier de crédit
feat_val_per_cust_2 = feat_val_per_cust.merge(feat_imp,
                                              left_on='index', right_on='feature').sort_values('importance',ascending=False)
feat_val_per_cust_2 = feat_val_per_cust_2[['feature','value','importance']]
feat_val_per_cust_2 = round(feat_val_per_cust_2,2)

#####################################
### Dossiers de crédit similaires ###
#####################################

test_features_nn = test_features_with_id
test_features_nn.reset_index(drop=True, inplace=True)

credit_cur = test_features_nn[test_features_nn['SK_ID_CURR'] == str(credit_id_default_test)+'_'].iloc[0:1,1:]

scoring_multi_nn = nn_credit_test.kneighbors(credit_cur, 15, return_distance=False)
best_nn = pd.DataFrame(scoring_multi_nn)

gap_0 = 5

best_nn_ = best_nn.iloc[0:1,gap_0-4:gap_0].T
best_nn_ = best_nn_.set_index([0]).merge(test_features_nn, left_index=True, right_index=True)
scoring_multi_nn = best_nn_.merge(scoring_test, left_on='SK_ID_CURR', right_on='SK_ID_CURR')[['SK_ID_CURR','_P','_Pred','_True']]

##################################################
### Dossiers de crédit similaires - base score ###
##################################################

current_P = scoring_test[scoring_test['SK_ID_CURR'] == str(credit_id_default_test)+'_'][['_P']].iloc[0,0]
gap_1 = current_P * gap_0 / 100
scoring_multi = scoring_test[((scoring_test['_P'] > (current_P - gap_1))
                              & (scoring_test['_P'] < current_P))| ((scoring_test['_P'] < (current_P + gap_1))
                                                                    & (scoring_test['_P'] > current_P))].sort_values('_P')

scoring_multi = pd.concat([scoring_multi.head(2),scoring_multi.tail(2)])
del scoring_multi['_N']

#############################################
### Choix du modèle de crédits similaires ###
#############################################

scoring_multi_ = scoring_multi

feat_val_per_cust_multi = scoring_multi_.merge(test_features_with_id, left_on='SK_ID_CURR',
                                              right_on='SK_ID_CURR').sort_values('_P')

feat_val_per_cust_multi.sort_index(inplace=True)
feat_val_per_cust_multi = feat_val_per_cust_multi.T

feat_val_per_cust_cur = test_features_with_id[test_features_with_id['SK_ID_CURR'] == str(credit_id_default_test)+'_']
feat_val_per_cust_cur = scoring_test.merge(feat_val_per_cust_cur, left_on='SK_ID_CURR',
                                              right_on='SK_ID_CURR').sort_values('_P')

feat_val_per_cust_cur = feat_val_per_cust_cur.T

feat_val_per_cust_multi = feat_val_per_cust_cur.merge(feat_val_per_cust_multi, left_index=True, right_index=True, how = 'right')
feat_val_per_cust_multi.reset_index(inplace=True)

tab_labels = ['Variable', 'Courant', 'Réf. 1', 'Réf. 2', 'Réf. 3', 'Réf. 4', 'Importance']
feat_imp.columns = ['index','importance']

# On fusionne les variables importantes avec les variables du dossier de crédit
feat_val_per_cust_2_2 = feat_val_per_cust_multi.merge(feat_imp,
                                                      left_on='index',
                                                      right_on='index',
                                                      how='left')

feat_val_per_cust_2_2 = round(feat_val_per_cust_2_2,2)
feat_val_per_cust_2_2.columns = tab_labels

#########################################
### Comparaison avec un autre dossier ###
#########################################

# Valeur des variables pour un individu (demande/dossier de crédit)
# Initialisation de la requête
feat_val_per_cust_1 = test_features_with_id[test_features_with_id['SK_ID_CURR'] == str(credit_id_default_test)+'_'].T
feat_val_per_cust_1.columns = ['courant']
feat_val_per_cust_1.reset_index(inplace=True)

feat_val_per_cust_2 = test_features_with_id[test_features_with_id['SK_ID_CURR'] == str(credit_id_default_test)+'_'].T
feat_val_per_cust_2.columns = ['dossier X']
feat_val_per_cust_2.reset_index(inplace=True)

feat_val_multi = feat_val_per_cust_1.merge(feat_val_per_cust_2, left_on='index', right_on='index')
feat_val_multi = feat_val_multi.loc[1:,:]
feat_val_multi.reset_index(drop=True, inplace=True)

sim_list = []

for i in feat_val_multi.index:
    
    diff = abs(feat_val_multi.iloc[i,2] - feat_val_multi.iloc[i,1])
    
    if abs(feat_val_multi.iloc[i,2]) > abs(feat_val_multi.iloc[i,1]):
        sim = diff / feat_val_multi.iloc[i,2]    
    elif abs(feat_val_multi.iloc[i,2]) < abs(feat_val_multi.iloc[i,1]):
        sim = diff / feat_val_multi.iloc[i,1]    
    elif feat_val_multi.iloc[i,2] == feat_val_multi.iloc[i,1]:
        sim = 0
        
    sim = abs(round((100*sim),0))
    sim_list.append(sim)

feat_val_multi['sim'] = sim_list
feat_val_multi = feat_val_multi.merge(feat_imp, left_on='index', right_on='index')
feat_val_multi.head()
feat_val_multi.columns = ['Variable','Courant','Référence','Variation','Importance']

# Scoring test_sub
test_sub_pred_bin = (test_sub_pred > solvability_threshold_g_norm)
test_sub_pred_bin = np.array(test_sub_pred_bin > 0) * 1
test_sub_pred_bin = pd.DataFrame(test_sub_pred_bin)

scoring_sub = df_test_sub_ids.reset_index(drop=True).merge(pd.DataFrame(test_sub_pred), left_index=True, right_index=True)
scoring_sub = scoring_sub.merge(test_sub_pred_bin, left_index=True, right_index=True)
scoring_sub.columns = ['SK_ID_CURR','_P','_Pred']

scoring_sub['_N'] = 1 - scoring_sub['_P']
scoring_sub = scoring_sub[['SK_ID_CURR','_N','_P','_Pred']]
# La base test_sub ne contient pas d'étiquette
# On définit alors des valeurs par défaut sur la base d'un seuil = 0.2
test_sub_pred_bin_2 = (test_sub_pred >= 0.2)
test_sub_pred_bin_2 = np.array(test_sub_pred_bin_2 > 0) * 1
test_sub_pred_bin_2 = pd.DataFrame(test_sub_pred_bin_2)
test_sub_pred_bin_2.columns=['TARGET']

scoring_sub = scoring_sub.merge(test_sub_pred_bin_2.reset_index(drop=True), left_index=True, right_index=True)
scoring_sub.columns = ['SK_ID_CURR','_N','_P','_Pred','_True']

# Scoring test_sub - Mise à jour de la liste de valeurs de l'identifiant 'SK_ID_CURR' (dropdownlist)
scoring_sub = scoring_sub.sort_values('_P')
scoring_sub = round(scoring_sub,5)

list_test_sub_credit_ids = scoring_sub[(scoring_sub['_P'] >= 0) & (scoring_sub['_P'] <= 1)]
list_test_sub_credit_ids = list_test_sub_credit_ids.merge(df_test_sub_ids_, left_on = 'SK_ID_CURR',
                                                          right_on = 'SK_ID_CURR', how='left')

list_test_sub_credit_ids = list_test_sub_credit_ids['SK_ID_CURR_num']

# Dataframe du meilleur et du moins bon résultat
# test
best_record = scoring_test.sort_values('_P').head(1)
worst_record = scoring_test.sort_values('_P').tail(1)
best_worst_records = pd.concat([best_record,worst_record])

# test_sub
best_record_sub = scoring_sub.sort_values('_P').head(1)
worst_record_sub = scoring_sub.sort_values('_P').tail(1)
best_worst_records_sub = pd.concat([best_record_sub,worst_record_sub])

# Identification de l'indice du dossier par défaut
credit_id = credit_id_default_test
credit_res = scor_test[scor_test['SK_ID_CURR'] == credit_id]
credit_index = credit_res.index[0]

############
### Shap ###
############

if heroku_deploy == 0:
    
    import shap

    explainerModel = joblib.load('../OC_DS_P7/interpretation/df_app_2_explainerModel.joblib')
    shap_values_Model = joblib.load('../OC_DS_P7/interpretation/df_app_2_shap_values_Model.joblib')

    j = credit_index
    S = test_features_ori

    # Initialisation du notebook avec initjs()
    shap.initjs()

    shap.force_plot(explainerModel.expected_value[1], shap_values_Model[1][j], S.iloc[[j]], show=False, matplotlib=True)

    plt.savefig(image_filename)
    plt.savefig(image_filename_ref)

    credit_id = credit_id_default_test_sub
    credit_res = scor_test_sub[scor_test_sub['SK_ID_CURR'] == credit_id]
    credit_index = credit_res.index[0]

    explainerModel_sub = joblib.load('../OC_DS_P7/interpretation/df_app_2_explainerModel_sub.joblib')
    shap_values_Model_sub = joblib.load('../OC_DS_P7/interpretation/df_app_2_shap_values_Model_sub.joblib')

########################
### Fonction globale ###
########################

def metrics(recall, x, data_to_pred):
    idx = np.max(np.where(thr >= recall))
    
    if recall == 0:
        thr_ch = 0
    else:
        thr_ch = round(thr[idx],5)
        
    recall = round(recall,5)
    
    # test
    test_pred_bin = (test_pred >= recall)
    test_pred_bin = np.array(test_pred_bin > 0) * 1
    test_pred_bin = pd.DataFrame(test_pred_bin)

    scoring = df_test_ids.reset_index(drop=True).merge(pd.DataFrame(test_pred), left_index=True, right_index=True)
    scoring = scoring.merge(test_pred_bin, left_index=True, right_index=True)
    scoring.columns = ['SK_ID_CURR','_P','_Pred']

    scoring['_N'] = 1 - scoring['_P']
    scoring = scoring[['SK_ID_CURR','_N','_P','_Pred']] 
    scoring = scoring.merge(pd.DataFrame(test_labels).reset_index(drop=True), left_index=True, right_index=True)
    scoring.columns = ['SK_ID_CURR','_N','_P','_Pred','_True']
    
    # Prédiction (0 ou 1)
    res_sco_test = scoring[scoring['SK_ID_CURR']==x]['_Pred'].values.tolist()
    # Etiquette (0 ou 1)
    lab_sco_test = scoring[scoring['SK_ID_CURR']==x]['_True'].values.tolist()

    # test_sub
    test_sub_pred_bin = (test_sub_pred >= recall)
    test_sub_pred_bin = np.array(test_sub_pred_bin > 0) * 1
    test_sub_pred_bin = pd.DataFrame(test_sub_pred_bin)

    scoring_sub = df_test_sub_ids.reset_index(drop=True).merge(pd.DataFrame(test_sub_pred), left_index=True, right_index=True)
    scoring_sub = scoring_sub.merge(test_sub_pred_bin, left_index=True, right_index=True)
    scoring_sub.columns = ['SK_ID_CURR','_P','_Pred']

    scoring_sub['_N'] = 1 - scoring_sub['_P']
    scoring_sub = scoring_sub[['SK_ID_CURR','_N','_P','_Pred']]

    test_sub_pred_bin_2 = (test_sub_pred >= 0.5) # Ratios avec seuil
    test_sub_pred_bin_2 = np.array(test_sub_pred_bin_2 > 0) * 1
    test_sub_pred_bin_2 = pd.DataFrame(test_sub_pred_bin_2)
    test_sub_pred_bin_2.columns=['TARGET']

    scoring_sub = scoring_sub.merge(test_sub_pred_bin_2.reset_index(drop=True), left_index=True, right_index=True)
    scoring_sub.columns = ['SK_ID_CURR','_N','_P','_Pred','_True']
    
    # Prédiction (0 ou 1)    
    res_sco_sub = scoring_sub[scoring_sub['SK_ID_CURR']==x]['_Pred'].values.tolist()
    # Etiquette (0 ou 1) - valeur définie sur la base arbitraire d'un seuil de 0.5
    lab_sco_sub = scoring_sub[scoring_sub['SK_ID_CURR']==x]['_True'].values.tolist()
    
    if data_to_pred == "Subm":
        res_sco = res_sco_sub # Prédiction
        lab_sco = lab_sco_sub # Valeur réelle
    
    elif data_to_pred == "Test":
        res_sco = res_sco_test
        lab_sco = lab_sco_test      

    ####################################################
    ### CALCUL DU GAIN (en fonction du seuil choisi) ###
    ####################################################

    # Identification, dans le dataframe df_g_norm, de la valeur de seuil la plus proche du seuil choisi 
    # dans l'objectif d'identifier le gain en rapport

    threshold_ch = recall

    threshold_fit_list = []

    # Pour chaque valeur du seuil
    for i in df_g_norm['seuil']:
        # On calcule la différence entre les valeurs de seuils
        threshold_fit = threshold_ch - i
        # On ajoute la différence à la liste
        threshold_fit_list.append(abs(threshold_fit))

    # On identifie la différence la plus petite (en valeur absolue)
    best_threshold_fit = min(threshold_fit_list)
    # On identifie le gain correspondant
    id_ = threshold_fit_list.index(best_threshold_fit)
    g_norm_ch = df_g_norm['g_norm'].loc[id_]

    ############################
    ### MATRICE DE CONFUSION ###
    ############################

    test_true = pd.DataFrame(test_labels).reset_index(drop=True)

    scoring_vs_true = scoring.merge(test_true, left_index=True, right_index=True)
    scoring_vs_true = scoring_vs_true.sort_index()

    x_pred = scoring_vs_true['_Pred']
    y_true = scoring_vs_true['TARGET']

    tn, fp, fn, tp = confusion_matrix(y_true, x_pred).ravel()   
    
    #################
    ### Métriques ###
    #################
    
    sensibilite = round(tpr[idx],2)
    specificite = round(1-fpr[idx],2)
    precision = round((tp / (tp + fp)),2)
    f_mesure = round((2*tp / (2*tp + fp + fn)),2)
    g_norm_ch = round(g_norm_ch,3)
    
    #############
    ### POLAR ###
    #############

    df = pd.DataFrame(dict(r=[precision, f_mesure, specificite, sensibilite, g_norm_ch],
                           theta=['Précision','F1','Spécificité','Sensibilité','Gain']))
    
    fig_pol = px.line_polar(df, r='r', theta='theta', line_close=True)
    
    fig_pol.update_layout(plot_bgcolor=colors['background_1'],
                          paper_bgcolor='lightgrey',
                          font_color=colors['text_1'],
                          font_size=10,
                          margin=dict(l=20, r=20, t=20, b=20),
                          autosize=True,
                          height=180)
    
    ################
    ### PIE PRED ###
    ################
    
    if data_to_pred == "Subm":
        scoring_pie = scoring_sub
    
    elif data_to_pred == "Test":
        scoring_pie = scoring
        
    _0_count = scoring_pie.shape[0] - scoring_pie['_Pred'].sum()
    _1_count = scoring_pie['_Pred'].sum()

    df_pie = pd.DataFrame({
    'Classe': ['Solvables','Non Solvables'],
    'Nombre': [_0_count,_1_count]}, index=[0,1])
    
    if _0_count >= _1_count:
        fig_pie = px.pie(df_pie, values='Nombre', names='Classe',
                         color_discrete_sequence=['rgba(0, 153, 51, 0.8)', 'rgba(255, 51, 0, 0.8)'])
    
    elif _0_count < _1_count:
        fig_pie = px.pie(df_pie, values='Nombre', names='Classe',
                         color_discrete_sequence=['rgba(255, 51, 0, 0.8)','rgba(0, 153, 51, 0.8)'])
    
    fig_pie.update(layout_showlegend=False)
    
    fig_pie.update_traces(hoverinfo='label',
                          textinfo='percent',
                          textfont_size=11,
                          marker=dict(line=dict(color='lightgrey', width=3))
                         )

    fig_pie.update_layout(plot_bgcolor=colors['background_1'],
                          paper_bgcolor='lightgrey',
                          font_color=colors['text_1'],
                          font_size=9,
                          margin=dict(l=20, r=20, t=10, b=10),
                          autosize=True,
                          height=160,
                          title={
                              'text': "Ratios prédictions",
                              'y':1.0,
                              'x':0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
        
    ################
    ### PIE TRUE ###
    ################
    
    if data_to_pred == "Subm":
        scoring_pie = scoring_sub
    
    elif data_to_pred == "Test":
        scoring_pie = scoring

    _0_count = scoring_pie.shape[0] - scoring_pie['_True'].sum()
    _1_count = scoring_pie['_True'].sum()

    df_pie = pd.DataFrame({
    'Classe': ['Solvables','Non Solvables'],
    'Nombre': [_0_count,_1_count]}, index=[0,1])
    
    if _0_count >= _1_count:
        fig_pie_true = px.pie(df_pie, values='Nombre', names='Classe',
                              color_discrete_sequence=['rgba(0, 153, 51, 0.8)', 'rgba(255, 51, 0, 0.8)'])
    
    elif _0_count < _1_count:
        fig_pie_true = px.pie(df_pie, values='Nombre', names='Classe',
                              color_discrete_sequence=['rgba(255, 51, 0, 0.8)','rgba(0, 153, 51, 0.8)'])
    
    fig_pie_true.update(layout_showlegend=False)
    
    fig_pie_true.update_traces(hoverinfo='label',
                               textinfo='percent',
                               textfont_size=11,
                               marker=dict(line=dict(color='lightgrey', width=3))
                              )

    if data_to_pred == "Subm":
        fig_pie_true.update_layout(plot_bgcolor=colors['background_1'],
                               paper_bgcolor='lightgrey',
                               font_color=colors['text_1'],
                               font_size=9,
                               margin=dict(l=20, r=20, t=20, b=10),
                               autosize=True,
                               height=160,
                               title={
                                   'text': "Ratios avec seuil = 0.5",
                                   'y':1,
                                   'x':0.5,
                                   'xanchor': 'center',
                                   'yanchor': 'top'})
    
    elif data_to_pred == "Test":
        fig_pie_true.update_layout(plot_bgcolor=colors['background_1'],
                               paper_bgcolor='lightgrey',
                               font_color=colors['text_1'],
                               font_size=9,
                               margin=dict(l=20, r=20, t=10, b=10),
                               autosize=True,
                               height=160,
                               title={
                                   'text': "Ratios réels",
                                   'y':1.0,
                                   'x':0.5,
                                   'xanchor': 'center',
                                   'yanchor': 'top'})
   
    ###############
    ### HEATMAP ###
    ###############

    z = [[fp, tp],
         [tn, fn]]
    
    x = ['Réel N', 'Réel P']
    y =  ['P', 'N']

    z_text = [[str(y) for y in x] for x in z]

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text,
                                      colorscale='Viridis')

    fig.update_layout(
                      margin=dict(t=10, l=20),
                      autosize=True,
                      font_size=11,
                      width=260,
                      height=170,
                      paper_bgcolor='lightgrey')

    fig['data'][0]['showscale'] = False    

    return auc_val,round(recall,5),round(tpr[idx],2),round(1-fpr[idx],2),res_sco,g_norm_ch,precision,f_mesure,fig,fig_pol,fig_pie,fig_pie_true,lab_sco


########################
### Application Dash ###
########################

# -*- coding: utf-8 -*-

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


# Graduation des Sliders
# Courbe ROC
marks = np.arange(0,1.1,0.1)
df_marks = pd.DataFrame(marks)
df_marks.columns = ['marks']
df_marks = round(df_marks,2)
# Gain
marks_2 = np.arange(0,1,0.02)
df_marks_2 = pd.DataFrame(marks_2)
df_marks_2.columns = ['marks']
df_marks_2 = round(df_marks_2,2)

# Fonction de création d'une bannière
def build_banner(banner_id,image_id,title_id,logo_id,title):
    return html.Div(
        id=banner_id,
        className=banner_id,
        children=[
            html.Div(className='row tabs_div',
                 children=[
                     html.Div(className='three columns',
                              id=image_id,
                              children=[
                                  html.Img(id=logo_id,src=app.get_asset_url('logo_pad.png'),height='90',width='90')
                              ]
                             ),
                     html.Div(className='eight columns',
                              id=title_id,
                              children=[
                                  html.H2(children=title,style={'textAlign': 'right','color': colors['text_1']}),
                              ]
                             )
                 ]
                    )
        ]
    )


def b64_image(image_filename):    
    with open(image_filename, 'rb') as f: image = f.read()    
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

def b64_image(image_filename_ref):    
    with open(image_filename_ref, 'rb') as f: image = f.read()    
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


###################
### Dash layout ###
###################

# L'option suivante consiste à placer le layout dans une fonction
# afin de mettre à jour le layout à chaque rafraîchissement de la page
# par défaut, Dash garde en mémoire le layout au premier chargement de la page

# def serve_layout(): return html.Div( # conteneur principal

app.layout = html.Div( # conteneur principal
    children=[
        html.Div(className='row tabs_div', # L1 (ligne 1)
                 children=[
                     
                     html.Div(), # L1C0 (ligne 1, colonne 0)
                     
                     html.Div(className='three columns', # L1C1 (ligne 1, colonne 1)
                              children=[
                                  build_banner("banner","banner-logo","banner-text","logo","Crédit Scoring"),
                                  html.H6(children='Dossier de crédit',
                                          style={'textAlign': 'left','color': colors['text_1']}),
                                  
                                  html.Div(className='row tabs_div', # L1C1-L1
                                           children=[
                                               html.Div(className='six columns', # L1C1-L1C1
                                                        children=[
                                                            dcc.Dropdown(
                                                                id='credit-id',
                                                                # Les options sont définies par la fonction set_input_options
                                                                value=credit_id_default_test # Valeur par défaut
                                                                                             # du numéro de dossier
                                                            ),
                                                        ]
                                                       ),
                                               html.Div(className='six columns', # L1C1-L1C2
                                                        children=[
                                                            dcc.RadioItems(
                                                                id='data-to-pred',
                                                                options=[{'label': i, 'value': i} for i in ['Test', 'Subm']],
                                                                value='Test',
                                                                labelStyle={'display': 'inline-block'}
                                                            ),
                                                        ],style={'textAlign': 'center'}
                                                       )
                                           ]
                                          ), # fin L1C1-L1

                                  html.Div(className='row tabs_div', # L1C1-L2
                                           children=[
                                               html.Div(className='four columns', # L1C1-L2C1
                                                        children=[
                                                            html.Table([
                                                                html.Tr([html.Td(['Seuil']), html.Td(id='seuil')])
                                                            ])
                                                        ]
                                                       ),
                                               html.Div(className='three columns', # L1C1-L2C2
                                                        children=[
                                                            html.Table([
                                                                html.Tr([html.Td(['Cible']), html.Td(id='target')])
                                                            ])
                                                        ]
                                                       ),
                                               html.Div(className='three columns', # L1C1-L2C2
                                                        children=[
                                                            html.Table([
                                                                html.Tr([html.Td(['Prédiction']), html.Td(id='pred_ts')])
                                                            ])
                                                        ]
                                                       )                                               
                                           ]
                                          ), # fin L1C1-L2
                                  
                                  html.Br(),
                                  dcc.Graph(id='fig-gauge')
                              ]
                             ), # fin L1C1
                     
                     html.Div(className='three columns', # L1C2 (ligne 1, colonne 2)
                              children=[
                                  html.H6(children='Indicateurs',style={'textAlign': 'left','color': colors['text_1']}),
                                  html.Div(className='row tabs_div', # L1C2-L1
                                           children=[
                                               html.Div(className='six columns', # L1C2-L1C1
                                                        children=[
                                                            html.Table([
                                                                html.Tr([html.Td(['AUC']), html.Td(id='auc')]),
                                                                html.Tr([html.Td(['Gain']), html.Td(id='gain')]),
                                                                html.Tr([html.Td(['F-Mesure']), html.Td(id='f_mesure')])
                                                            ])
                                                        ]
                                                       ),
                                               html.Div(className='five columns', # L1C2-L1C2
                                                        children=[
                                                            html.Table([
                                                                
                                                                html.Tr([html.Td(['Sensibilité']), html.Td(id='sensibilite')]),
                                                                html.Tr([html.Td(['Spécificité']), html.Td(id='specificite')]),
                                                                html.Tr([html.Td(['Précision']), html.Td(id='precision')])
                                                            ])
                                                        ]
                                                       )
                                           ]
                                          ), # fin L1C2-L1                               
                                  
                                  html.Br(),
                                  html.Br(),
                                  html.Div(className='row tabs_div', # L1C2-L2
                                           children=[
                                               html.Div(className='six columns', # L1C2-L2C1
                                                        children=[
                                                            dcc.Graph(id='pie'),
                                                        ]
                                                       ),
                                               html.Div(className='six columns', # L1C2-L2C2
                                                        children=[
                                                            dcc.Graph(id='pie_true'),
                                                        ],style={'textAlign': 'center'}
                                                       )
                                           ]
                                          ), # fin L1C2-L2
                              ]
                             ), # fin L1C2
                     
                     html.Div(className='two columns', # L1C3 (ligne 1, colonne 3)
                              children=[
                                  html.Br(),
                                  dcc.Graph(id='polar'),
                                  html.Button(id='submit-button-state', n_clicks=1, children='Seuil optimal'),
                                  dcc.Graph(id='heatmap')
                              ],style={'textAlign': 'center'}
                             ), # fin L1C3
                     
                     html.Div(className='three columns', # L1C4 (ligne 1, colonne 4)
                              children=[
                                  html.H6(children='Simulation seuil',style={'textAlign': 'left','color': colors['text_1']}),
                                  dcc.Graph(id='simulation-roc'),
                                  html.Br(),
                                  html.Div(id='slider-output-container'),
                                  # Slider avec step
                                  dcc.Slider(
                                          id='slider-threeshold-c',
                                          min=0,
                                          max=1,
                                          step=0.01,
                                          value=solvability_threshold_g_norm,
                                      ),
                              ]
                             ) # fin L1C4
                 ]
                ), # fin L1
        
        html.Div(className='row', # L2 (ligne 2)
                 children=[
                     
                     html.Div(), # L2C0 (ligne 2, colonne 0)
                     
                     html.Div(className='three columns', # L2C1 (ligne 2, colonne 1)
                              children=[
                                  html.H6(children='Score',style={'textAlign': 'left','color': colors['text_1']}),
                                  html.Div(children='''
                                  Le score du dossier est la probabilité que le crédit ne soit pas remboursé.
                                  ''',style={'textAlign': 'left', 'color': colors['text_4']}),
                                  html.Br(),
                                  dcc.Graph(id='fig-scoring'),
                                  
                                  html.Div(children='''
                                  Le score du dossier en cours est situé entre le meilleur score (plus petit)
                                  et le moins bon score (plus grand).
                                  ''',style={'textAlign': 'left', 'color': colors['text_4']}),
                                  html.Br(),
                                  
                                  build_banner("banner_","banner-logo_","banner-text_","logo_","Interprétabilité")
                              ]
                             ), # fin L2C1
                     
                     html.Div(className='three columns', # L2C2 (ligne 2, colonne 2)
                              children=[
                                  html.H6(children='Analyse statistique',
                                          style={'textAlign': 'left','color': colors['text_1']}),
                                  
                                  html.Div(children='''
                                  Le seuil de classification permet de classer un dossier selon son score.
                                  Si le score est supérieur au seuil, le dossier présente des risques.
                                  ''',style={'textAlign': 'left', 'color': colors['text_4']}),
                                  html.Br(),
                                  
                                  html.Div(children='''
                                  On peut modifier le statut du dossier en faisant varier le seuil
                                  et observer les conséquences sur d'autres indicateurs
                                  avec les simulations de seuil et de gain.
                                  ''',style={'textAlign': 'left', 'color': colors['text_4']})
                              ]
                             ), # fin L2C2
                     
                     html.Div(className='two columns', # L2C3 (ligne 2, colonne 3)
                              children=[
                                  html.Br(),
                                  dcc.Graph(id='heatmap-2'),
                                  dcc.Graph(id='heatmap-3')
                              ],style={'textAlign': 'center'}
                             ), # fin L2C3
                     
                     html.Div(className='three columns', # L2C4 (ligne 2, colonne 4)
                              children=[
                                  html.H6(children='Simulation gain',style={'textAlign': 'left','color': colors['text_1']}),
                                  dcc.Graph(id='simulation-gain'),
                                  html.Br(),
                                  html.Div(id='slider-output-container-g'),
                                  # Slider avec step
                                  dcc.Slider(
                                          id='slider-gain-c',
                                          min=df_g_norm['g_norm'].min(),
                                          max=df_g_norm['g_norm'].max()+0.01,
                                          step=0.001,
                                          value=0.65
                                      ),
                              ]
                             ) # fin L2C4
                 ]
                ), # fin L2
        
        html.Div(className='row', # L3 (ligne 3)
                 children=[
                     
                     html.Div(), # L3C0 (ligne 3, colonne 0)
                     
                     html.Div(className='five columns', # L3C1 (ligne 3, colonne 1)
                              children=[
                                  
                                  html.H6(children='Variables',style={'textAlign': 'left','color': colors['text_1']}),
                                  html.Div(children='''
                                  Les informations des dossiers "courant" et de "référence" sont affichées avec
                                  leur taux de variation et leur importance normalisée (0 à 100).
                                  ''',style={'textAlign': 'left', 'color': colors['text_4']}),
                                  html.Br(),
                                  
                                  html.Div(className='row tabs_div', # L3C1-L1
                                           children=[
                                               html.Div(className='four columns', # L3C1-L1C1
                                                        children=[
                                                            html.Div(id='credit-id-info')
                                                        ]
                                                       ),
                                               html.Div(className='four columns', # L3C1-L1C2
                                                        children=[
                                                            html.Div('Dossier de référence:')
                                                        ],style={'textAlign': 'right'}
                                                       ),
                                               html.Div(className='four columns', # L3C1-L1C3
                                                        children=[
                                                            dcc.Dropdown(
                                                                id='credit-id-2',
                                                                # Les options sont définies par la fonction set_input_options
                                                                value= credit_id_default_test # Valeur par défaut
                                                                                              # du numéro de dossier
                                                            ),
                                                        ],style={'textAlign': 'right'}
                                                       )
                                           ]
                                          ), # fin L3C1-L1
                                  
                                  html.Br(),
                                  dash_table.DataTable(
                                      id='table-paging-with-graph',
                                      columns=[
                                          {"name": i, "id": i} for i in (feat_val_multi.columns)
                                      ],
                                      page_current=0,
                                      page_size=10, # Nombre de lignes par page
                                      page_action='custom',
                                      filter_action='custom',
                                      filter_query='',
                                      sort_action='custom',
                                      sort_mode='multi',
                                      sort_by=[]
                                  )
                              ]
                             ), # fin L3C1
                     
                     html.Div(className='five columns', # L3C2 (ligne 3, colonne 2)
                              children=[
                                  html.H6(children='Analyse métier',style={'textAlign': 'left','color': colors['text_1']}),
                                  html.Div(children='''
                                  Le dossier "courant" est comparé avec des dossiers similaires, sur la base du score
                                  ou des variables.
                                  ''',style={'textAlign': 'left', 'color': colors['text_4']}),
                                  html.Br(),
                                  
                                  html.Div(className='row tabs_div', # L3C2-L1
                                           children=[
                                               html.Div(className='four columns', # L3C2-L1C1
                                                        children=[
                                                            dcc.RadioItems(
                                                                id='sim-mod',
                                                                options=[{'label': i, 'value': i} for i in ['Score','Variables']],
                                                                value='Score',
                                                                labelStyle={'display': 'inline-block'}
                                                            ),
                                                        ]
                                                       ),
                                               html.Div(className='three columns', # L3C2-L1C2
                                                        children=[
                                                            html.Div("Degré de similarité:")
                                                        ],style={'textAlign': 'right'}
                                                       ),
                                               html.Div(className='five columns', # L3C2-L1C3
                                                        children=[
                                                            html.Div([dcc.Input(id='similarity-gap', value=5,
                                                                                type='number', size=10)])
                                                        ],style={'textAlign': 'right'}
                                                       )
                                           ]
                                          ), # fin L3C2-L1

                                  html.Br(),
                                  dash_table.DataTable(
                                      id='table-paging-with-graph-2',
                                      columns=[
                                          {"name": i, "id": i} for i in (feat_val_per_cust_2_2.columns)
                                      ],
                                      page_current=0,
                                      page_size=10, # Nombre de lignes par page
                                      page_action='custom',
                                      filter_action='custom',
                                      filter_query='',
                                      sort_action='custom',
                                      sort_mode='multi',
                                      sort_by=[]
                                  )
                              ]
                             ), # fin L3C2
                     
                     html.Div(className='two columns', # L3C3 (ligne 3, colonne 3)
                              children=[
                              ]
                             ), # fin L3C3
                     
                     html.Div(className='two columns', # L3C4 (ligne 3, colonne 4)
                              children=[
                              ]
                             ), # fin L3C4
                 ]
                ), # fin L3        
        
        html.Div(className='row', # L4 (ligne 4)
                 children=[
                     html.Div(), # L4C0 (ligne 4, colonne 0)
                     
                     html.Div(className='five columns', # L4C1 (ligne 4, colonne 1)
                              children=[
                                  html.H6(children='Interprétation par dossier',
                                          style={'textAlign': 'left','color': colors['text_1']}),
                                  # Affichage de l'image shap_force_plot.png
                                  html.Div(id='shap-ima-info'),
                                  html.Img(id="body-image",
                                           src=b64_image(image_filename), height='190', width='900'),
                                  html.Br(),
                                  # Affichage de l'image shap_force_plot_ref.png
                                  html.Div(id='shap-ima-info-ref'),
                                  html.Img(id="body-image-ref",
                                           src=b64_image(image_filename_ref), height='190', width='900'),
                                  # Affichage de la date/heure de mise à jour
                                  # La mise à jour est déclenchée après modification du numéro de dossier
                                  html.Div(id='ima-time'),
                                  html.Br(),
                                  html.Div('Connexion: ' + str(datetime.datetime.now()))
                              ]
                             ), # fin L4C1
                     
                     html.Div(className='five columns', # L4C2 (ligne 4, colonne 2)

                              children=[
                              ]
                             ), # fin L4C2
                     
                     html.Div(className='two columns', # L4C3 (ligne 4, colonne 3)
                              children=[
                              ]
                             ), # fin L4C3
                     
                     html.Div(className='two columns', # L4C4 (ligne 4, colonne 4)
                              children=[
                              ]
                             ), # fin L4C4
                 ]
                ) # fin L4

    
    ],style={'background-color': 'lightgrey'}
    
) # fin layout


# On active l'option suivante dans le cas où le layout est
# placé dans la fonction def serve_layout()

# app.layout = serve_layout

operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part
                        
                return name, operator_type[0].strip(), value

    return [None] * 3


######################
### Dash callbacks ###
######################

# Seuil optimal
@app.callback(
    Output('slider-threeshold-c', 'value'),
    [Input('submit-button-state', 'n_clicks')])

def update_scoring(n_clicks):
    return solvability_threshold_g_norm

if heroku_deploy == 0:

    # Affichage des images shap
    @app.callback(
        [Output('shap-ima-info', 'children'),
         Output('body-image', 'src'),
         Output('ima-time', 'children'),
         Output('body-image-ref', 'src'),
         Output('shap-ima-info-ref', 'children')],
        [Input(component_id='credit-id', component_property='value'),
         Input('data-to-pred', 'value'),
         Input(component_id='credit-id-2', component_property='value'),])

    def shap(x, data_to_pred, x_ref):

        import shap
        import shutil, os

        if data_to_pred == "Subm":
            
            # Création de l'image shap pour le dossier courant  
            credit_res = scor_test_sub[scor_test_sub['SK_ID_CURR'] == x]
            credit_index = credit_res.index[0]
            shap_ima_info = 'Dossier courant: '+str(x)+' (indice = '+str(credit_index)+')'

            j = credit_index
            S = test_sub_features_ori

            shap.force_plot(explainerModel_sub.expected_value[1], shap_values_Model_sub[1][j],
                            S.iloc[[j]], show=False, matplotlib=True)

            plt.savefig(image_filename)
            src = b64_image(image_filename)
            ima_time = time.ctime(os.path.getmtime(image_filename))

            # Création de l'image shap pour le dossier de référence        
            credit_res_ref = scor_test_sub[scor_test_sub['SK_ID_CURR'] == x_ref]
            credit_index_ref = credit_res_ref.index[0]
            shap_ima_info_ref = 'Dossier de référence: '+str(x_ref)+' (indice = '+str(credit_index_ref)+')'

            j = credit_index_ref
            S = test_sub_features_ori

            shap.force_plot(explainerModel_sub.expected_value[1], shap_values_Model_sub[1][j],
                            S.iloc[[j]], show=False, matplotlib=True)

            plt.savefig(image_filename_ref)
            src_ref = b64_image(image_filename_ref)

        elif data_to_pred == "Test":
            
            # Création de l'image shap pour le dossier courant
            credit_res = scor_test[scor_test['SK_ID_CURR'] == x]
            credit_index = credit_res.index[0]
            shap_ima_info = 'Dossier courant: '+str(x)+' (indice = '+str(credit_index)+')'

            j = credit_index
            S = test_features_ori

            shap.force_plot(explainerModel.expected_value[1], shap_values_Model[1][j],
                            S.iloc[[j]], show=False, matplotlib=True)

            plt.savefig(image_filename)
            src = b64_image(image_filename)
            ima_time = time.ctime(os.path.getmtime(image_filename))

            # Création de l'image shap pour le dossier de référence        
            credit_res_ref = scor_test[scor_test['SK_ID_CURR'] == x_ref]
            credit_index_ref = credit_res_ref.index[0]
            shap_ima_info_ref = 'Dossier de référence: '+str(x_ref)+' (indice = '+str(credit_index_ref)+')'

            j = credit_index_ref
            S = test_features_ori

            shap.force_plot(explainerModel.expected_value[1], shap_values_Model[1][j],
                            S.iloc[[j]], show=False, matplotlib=True)

            plt.savefig(image_filename_ref)
            src_ref = b64_image(image_filename_ref)

        return shap_ima_info, src, 'Mise à jour: {}'.format(ima_time), src_ref, shap_ima_info_ref


@app.callback(
    [Output('credit-id', 'options'),
     Output('credit-id-2', 'options'),
     Output('credit-id', 'value'),
     Output('credit-id-2', 'value')],
    [Input('data-to-pred', 'value')])

def set_input_options(data_to_pred):
    
    credit_list_test = [{'label': i, 'value': i} for i in list_test_credit_ids]
    credit_list_sub = [{'label': i, 'value': i} for i in list_test_sub_credit_ids]
    
    if data_to_pred == 'Subm':
        return credit_list_sub, credit_list_sub, credit_id_default_test_sub, credit_id_default_test_sub
    elif data_to_pred == 'Test':
        return credit_list_test, credit_list_test, credit_id_default_test, credit_id_default_test

    
@app.callback(
    [dash.dependencies.Output('slider-output-container', 'children'),
     Output('simulation-roc', 'figure')],
#     Output('simulation-roc', 'figure'),
    [dash.dependencies.Input('slider-threeshold-c', 'value')])

def update_roc(thr_ch):

    idx = np.max(np.where(thr > thr_ch))
    x0 = df_roc['fpr']
    x1 = [0, fpr[idx]]
    x2 = [fpr[idx], fpr[idx]]

    fig_roc = go.Figure()    
    
    # Aire AUC
    fig_roc.add_trace(go.Scatter(
        x=x0,
        y=df_roc['tpr'],
        name='AUC',
        mode = 'lines',
        fill='tozeroy',
        line=dict(width=1),
        line_color='lightgrey'
    ))
    
    # Courbe ROC
    fig_roc.add_trace(go.Scatter(
        x=x0,
        y=df_roc['tpr'],
        name='<b>ROC</b>',
        mode = 'lines',
        line=dict(width=3),
        line_color='rgba(0, 153, 51, 0.6)'
    ))

    # Sensibilité
    fig_roc.add_trace(go.Scatter(
        x=x1,
        y=[tpr[idx],tpr[idx]],
        line_color='rgba(255, 51, 0, 0.8)',
        name = 'Sensibilité',
        connectgaps=True
    ))

    # Antispécificité
    fig_roc.add_trace(go.Scatter(
        x=x2,
        y=[tpr[idx],0],
        line_color='rgba(0, 102, 153, 0.6)',
        name='1 - Spécificité',
    ))

    fig_roc.update_layout(
        
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=True,
            zerolinecolor='rgba(255, 51, 0, 0.8)'
        ),
        
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=True,
            zerolinecolor='rgba(0, 102, 153, 0.6)'
        ),
               
        plot_bgcolor=colors['background_2'],
        paper_bgcolor=colors['background_2'],
        font_color=colors['text_1'],
        margin=dict(l=20, r=20, t=40, b=20),
        autosize=True,
        height=230
    )

    return 'Seuil sélectionné: {}'.format(thr_ch), fig_roc
#     return fig_roc


@app.callback(
    [dash.dependencies.Output('slider-output-container-g', 'children'),
     Output('simulation-gain', 'figure')],
    [dash.dependencies.Input('slider-gain-c', 'value')])

def update_gain(gain_ch):

    df_g_norm_filt = df_g_norm[df_g_norm['g_norm'] >= gain_ch]

    seuil_min = df_g_norm_filt[['g_norm','seuil']].sort_values('seuil').iloc[0]['seuil']
    gain_seuil_min = df_g_norm_filt[['g_norm','seuil']].sort_values('seuil').iloc[0]['g_norm']
    seuil_max = df_g_norm_filt[['g_norm','seuil']].sort_values('seuil', ascending=False).iloc[0]['seuil']
    gain_seuil_max = df_g_norm_filt[['g_norm','seuil']].sort_values('seuil', ascending=False).iloc[0]['g_norm']

    x0 = df_g_norm['seuil']

    x1 = [0, solvability_threshold_g_norm]
    x2 = [solvability_threshold_g_norm, solvability_threshold_g_norm]
    
    fig_gain = go.Figure()

    # Courbe d'évolution du gain
    fig_gain.add_trace(go.Scatter(
        x=x0,
        y=df_g_norm['g_norm'],
        fill='tozeroy',
        mode = 'none',
        name='<b>Gain</b>'
    ))

    # Marge de gain
    fig_gain.add_trace(go.Scatter(
        x=df_g_norm_filt['seuil'],
        y=df_g_norm_filt['g_norm'],
        fill='tozeroy', # Aire
        mode = 'lines',
        line=dict(width=3),
        name='Marge de gain',
        line_color='rgba(255, 51, 0, 0.6)'
    ))

    # Seuil optimal
    fig_gain.add_trace(go.Scatter(
        x=[solvability_threshold_g_norm,solvability_threshold_g_norm],
        y=[0,g_norm_opt],
        name='Seuil optimal',
        mode = 'lines+markers',
        line=dict(color='rgba(0, 153, 51, 0.6)',width=2)
    ))
    
    # Gain optimal
    fig_gain.add_trace(go.Scatter(
        x=x1,
        y=[g_norm_opt, g_norm_opt],
        name = 'Gain optimal',
        connectgaps=True,
        mode = 'lines+markers',
        line=dict(color='rgba(255, 51, 0, 0.6)',width=2)
    ))

    # Seuil min
    fig_gain.add_trace(go.Scatter(
        x=[seuil_min,seuil_min],
        y=[0, gain_seuil_min],
        name='Seuil min',
        mode='lines',
        line=dict(color='rgb(255,255,255)',width=1,dash='dash') # 'dash','dot','dashdot'
    ))

    # Seuil max
    fig_gain.add_trace(go.Scatter(
        x=[seuil_max,seuil_max],
        y=[0, gain_seuil_max],
        name='Seuil max',
        mode='lines',
        line=dict(color='rgb(0,0,0)',width=1,dash='dash')
    ))

    fig_gain.update_layout(
        
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=True,
            zerolinecolor='rgba(102, 102, 153, 0.5)'),
        
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=True,
            zerolinecolor='rgba(102, 102, 153, 0.5)'),
        
        plot_bgcolor=colors['background_2'],
        paper_bgcolor=colors['background_2'],
        font_color=colors['text_1'],
        font_size=11,
        margin=dict(l=20, r=20, t=40, b=20),
        autosize=True,
        height=210
    )

    return 'Gain sélectionné: {}'.format(gain_ch), fig_gain


@app.callback(
    [Output('fig-scoring', 'figure'),
     Output('fig-gauge', 'figure')],
    [Input(component_id='credit-id', component_property='value'),
     dash.dependencies.Input('slider-threeshold-c', 'value'),
     Input('data-to-pred', 'value')])

def update_scoring(input_value, thr_ch, data_to_pred):
    
    ###############
    ### Scoring ###
    ###############
    
    input_value_str = str(input_value) + '_'
    
    if data_to_pred == "Subm":
        scoring = scoring_sub
        best_record_ = best_record_sub
        worst_record_ = worst_record_sub
        best_worst_records_ = best_worst_records_sub
    
    elif data_to_pred == "Test":
        scoring = scoring_test
        best_record_ = best_record
        worst_record_ = worst_record
        best_worst_records_ = best_worst_records
    
    filtered_scoring_current = scoring[scoring['SK_ID_CURR'] == input_value_str]
    
    if input_value_str == best_record_['SK_ID_CURR'].iloc[0]:
        filtered_scoring = pd.concat([filtered_scoring_current,worst_record_])
        filtered_scoring = filtered_scoring.sort_values('_P', ascending=False)
    
    elif input_value_str == worst_record_['SK_ID_CURR'].iloc[0]:
        filtered_scoring = pd.concat([filtered_scoring_current,best_record_])
    
    else:
        filtered_scoring = pd.concat([filtered_scoring_current,best_worst_records_])
        filtered_scoring = filtered_scoring.sort_values('_P', ascending=False)
    
    credit = filtered_scoring['SK_ID_CURR']
    yp = round(filtered_scoring['_P'],5) # Probabilité non-solvabilité (positif: 0)
    yn = round(filtered_scoring['_N'],5) # Probabilité solvabilité (negatif: 1)
    y_pred = filtered_scoring['_Pred'] # Prédiction binaire (0 ou 1)

    fig_scoring = go.Figure()
    
    fig_scoring.add_trace(go.Bar(
        y=credit,
        x=yp,
        name='_P',
        orientation='h',
        text=yp,
        textposition='inside',
        marker=dict(
                color='rgba(255, 51, 0, 0.8)',
                line=dict(color='lightgrey', width=1)
        )
    ))
    
    fig_scoring.add_trace(go.Bar(
        y=credit,
        x=yn,
        name='_N',
        orientation='h',
        text=yn,
        textposition='inside',
        marker=dict(
            color='rgba(0, 153, 51, 0.8)',
            line=dict(color='lightgrey', width=1)
        )
    ))

    fig_scoring.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False,
        ),
        barmode='stack',
                      margin=dict(l=20, r=20, t=20, b=20),
                      autosize=True,
                      height=125,
                      plot_bgcolor=colors['background_2'],
                      paper_bgcolor=colors['background_2'],
                     )  

    fig_scoring.update_layout(transition_duration=500)
    
    #############
    ### Jauge ###
    #############
    
    if input_value_str == best_record_['SK_ID_CURR'].iloc[0]:
        proba_ = pd.DataFrame(yp).iloc[1,0]
        proba_worst_ = pd.DataFrame(yp).iloc[0,0]
    
    elif input_value_str == worst_record_['SK_ID_CURR'].iloc[0]:
        proba_ = pd.DataFrame(yp).iloc[0,0]
        proba_worst_ = pd.DataFrame(yp).iloc[0,0]
        
    else:
        proba_ = pd.DataFrame(yp).iloc[1,0]
        proba_worst_ = pd.DataFrame(yp).iloc[0,0]

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = proba_,
#         title = {'text': "Risque d'insolvabilité", 'font': {'size': 11}},
        delta = {'reference': thr_ch, 'increasing': {'color': "Red"}, 'decreasing': {'color': "Green"}},
        gauge = {
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': 'rgba(255, 255, 255, 0.6)'},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, thr_ch], 'color': 'rgba(0, 153, 51, 0.8)'},
                {'range': [thr_ch, 1], 'color': 'rgba(255, 51, 0, 0.8)'}],
            'threshold': {
                'line': {'color': "white", 'width': 1},
                'thickness': 1,
                'value': proba_worst_}}))

    fig_gauge.update_layout(paper_bgcolor = "lightgrey",
                            autosize=True,
                            height=150,
                            margin=dict(l=30, r=30, t=20, b=20),
                            font = {'color': "grey", 'family': "Arial"})

    return fig_scoring, fig_gauge


@app.callback(
    [Output('auc', 'children'),
     Output('seuil', 'children'),
     Output('sensibilite', 'children'),
     Output('specificite', 'children'),
     Output('pred_ts', 'children'),
     Output('gain', 'children'),
     Output('precision', 'children'),
     Output('f_mesure', 'children'),
     Output('heatmap', 'figure'),
     Output('target', 'children'),
     Output('credit-id-info', 'children')],
    [dash.dependencies.Input('slider-threeshold-c', 'value'),
     Input(component_id='credit-id', component_property='value'),
     Input('data-to-pred', 'value')])

def metrics_heatmap_1(thr_ch, x, data_to_pred):
    
    x_ = str(x) + '_'
    
    a,b,c,d,e,f,g,h,i,j,k,l,m = metrics(thr_ch, x_, data_to_pred)
    credit_id_info = 'Dossier courant: '+str(x)
    
    return a,b,c,d,e,f,g,h,i,m,credit_id_info
    

@app.callback(
    Output('heatmap-2', 'figure'),
    [dash.dependencies.Input('slider-gain-c', 'value'),
     Input(component_id='credit-id', component_property='value'),
     Input('data-to-pred', 'value')])

def heatmap_2(gain_ch, x, data_to_pred):
    
    x = str(x) + '_'
    
    df_g_norm_filt = df_g_norm[df_g_norm['g_norm'] >= gain_ch]
    seuil_min = df_g_norm_filt[['g_norm','seuil']].sort_values('seuil').iloc[0]['seuil']
    
    a,b,c,d,e,f,g,h,i,j,k,l,m = metrics(seuil_min, x, data_to_pred)
    
    return i


@app.callback(
    Output('heatmap-3', 'figure'),
    [dash.dependencies.Input('slider-gain-c', 'value'),
     Input(component_id='credit-id', component_property='value'),
     Input('data-to-pred', 'value')])

def heatmap_3(gain_ch, x, data_to_pred):
    
    x = str(x) + '_'
    
    df_g_norm_filt = df_g_norm[df_g_norm['g_norm'] >= gain_ch]
    seuil_max = df_g_norm_filt[['g_norm','seuil']].sort_values('seuil', ascending=False).iloc[0]['seuil']
    
    a,b,c,d,e,f,g,h,i,j,k,l,m = metrics(seuil_max, x, data_to_pred)
    
    return i


@app.callback(
    Output('polar', 'figure'),
    [dash.dependencies.Input('slider-threeshold-c', 'value'),
     Input(component_id='credit-id', component_property='value'),
     Input('data-to-pred', 'value')])

def fig_polar(thr_ch, x, data_to_pred):
    
    x = str(x) + '_'
 
    a,b,c,d,e,f,g,h,i,j,k,l,m = metrics(thr_ch, x, data_to_pred)
    
    return j


@app.callback(
    [Output('pie', 'figure'),
     Output('pie_true', 'figure')],
    [dash.dependencies.Input('slider-threeshold-c', 'value'),
     Input(component_id='credit-id', component_property='value'),
     Input('data-to-pred', 'value')])

def fig_pie(thr_ch, x, data_to_pred):
    
    x = str(x) + '_'
 
    a,b,c,d,e,f,g,h,i,j,k,l,m = metrics(thr_ch, x, data_to_pred)
    
    return k,l


@app.callback(
    Output('table-paging-with-graph', "data"),
    [Input('table-paging-with-graph', "page_current"),
     Input('table-paging-with-graph', "page_size"),
     Input('table-paging-with-graph', "sort_by"),
     Input('table-paging-with-graph', "filter_query"),
     Input(component_id='credit-id', component_property='value'),
     Input('data-to-pred', 'value'),
     Input(component_id='credit-id-2', component_property='value')])

def features_importance_tab(page_current, page_size, sort_by, filter, x, data_to_pred, x2):
    
    x = str(x) + '_'
    x2 = str(x2) + '_'
    
    if data_to_pred == "Subm":
        test_features_with_id_ = test_sub_features_with_id
    
    elif data_to_pred == "Test":
        test_features_with_id_ = test_features_with_id
        
    # Valeur des variables pour le dossier courant
    feat_val_per_cust_1 = test_features_with_id_[test_features_with_id_['SK_ID_CURR']==x].T
    feat_val_per_cust_1.columns = ['courant']
    feat_val_per_cust_1.reset_index(inplace=True)

    # Valeur des variables pour le dossier de référence
    feat_val_per_cust_2 = test_features_with_id_[test_features_with_id_['SK_ID_CURR']==x2].T
    feat_val_per_cust_2.columns = ['dossier X']
    feat_val_per_cust_2.reset_index(inplace=True)

    feat_val_multi = feat_val_per_cust_1.merge(feat_val_per_cust_2, left_on='index', right_on='index')

    feat_val_multi = feat_val_multi.loc[1:,:]
    feat_val_multi.reset_index(drop=True, inplace=True)

    sim_list = []

    for i in feat_val_multi.index:

        diff = abs(feat_val_multi.iloc[i,2] - feat_val_multi.iloc[i,1])

        if abs(feat_val_multi.iloc[i,2]) > abs(feat_val_multi.iloc[i,1]):
            sim = diff / feat_val_multi.iloc[i,2]
        elif abs(feat_val_multi.iloc[i,2]) < abs(feat_val_multi.iloc[i,1]):
            sim = diff / feat_val_multi.iloc[i,1]
        elif feat_val_multi.iloc[i,2] == feat_val_multi.iloc[i,1]:
            sim = 0

        sim = abs(round((100*sim),0))
        sim_list.append(sim)

    feat_val_multi['sim'] = sim_list
    feat_val_multi = feat_val_multi.merge(feat_imp, left_on='index', right_on='index')
    feat_val_multi.columns = ['Variable','Courant','Référence','Variation','Importance']
    
     
    filtering_expressions = filter.split(' && ')
    dff = feat_val_multi
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff = dff.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

    return dff.iloc[ 
        page_current*page_size: (page_current + 1)*page_size
    ].to_dict('records')


@app.callback(
    Output('table-paging-with-graph-2', "data"),
    [Input('table-paging-with-graph-2', "page_current"),
     Input('table-paging-with-graph-2', "page_size"),
     Input('table-paging-with-graph-2', "sort_by"),
     Input('table-paging-with-graph-2', "filter_query"),
     Input(component_id='credit-id', component_property='value'),
     Input('data-to-pred', 'value'),
     Input(component_id='similarity-gap', component_property='value'),
     Input('sim-mod', 'value'),])

def features_importance_tab_2(page_current, page_size, sort_by, filter, x, data_to_pred, gap_0, sim_mod):
    
    x = str(x) + '_'
    
    if data_to_pred == "Subm":
        test_features_with_id_ = test_sub_features_with_id
        scoring_test_ = scoring_sub
        nn_credit_test_ = nn_credit_test_sub
    
    elif data_to_pred == "Test":
        test_features_with_id_ = test_features_with_id
        scoring_test_ = scoring_test
        nn_credit_test_ = nn_credit_test
    
    if sim_mod == "Variables":
        test_features_nn = test_features_with_id_
        test_features_nn.reset_index(drop=True, inplace=True)

        credit_cur = test_features_nn[test_features_nn['SK_ID_CURR'] == x].iloc[0:1,1:]

        scoring_multi_nn = nn_credit_test_.kneighbors(credit_cur, 15, return_distance=False)
        best_nn = pd.DataFrame(scoring_multi_nn)

        best_nn_ = best_nn.iloc[0:1,gap_0-4:gap_0].T
        best_nn_ = best_nn_.set_index([0]).merge(test_features_nn, left_index=True, right_index=True)

        scoring_multi = best_nn_.merge(scoring_test_, left_on='SK_ID_CURR',
                                       right_on='SK_ID_CURR')[['SK_ID_CURR','_P','_Pred','_True']]
    
    elif sim_mod == "Score":
        current_P = scoring_test_[scoring_test_['SK_ID_CURR'] == x][['_P']].iloc[0,0]
        gap_1 = current_P * gap_0 / 100

        scoring_multi = scoring_test_[((scoring_test_['_P'] > (current_P - gap_1))
                                      & (scoring_test_['_P'] < current_P))| ((scoring_test_['_P'] < (current_P + gap_1))
                                                                            & (scoring_test_['_P'] > current_P))].sort_values('_P')

        scoring_multi = pd.concat([scoring_multi.head(2),scoring_multi.tail(2)])
        del scoring_multi['_N']

    
    feat_val_per_cust = scoring_multi.merge(test_features_with_id_, left_on='SK_ID_CURR',
                                            right_on='SK_ID_CURR').sort_values('_P')   
    feat_val_per_cust.sort_index(inplace=True)
    feat_val_per_cust = feat_val_per_cust.T    
    scoring_test_[scoring_test_['SK_ID_CURR']==x]    
    feat_val_per_cust_cur = test_features_with_id_[test_features_with_id_['SK_ID_CURR']==x]    
    feat_val_per_cust_cur = scoring_test_.merge(feat_val_per_cust_cur, left_on='SK_ID_CURR',
                                                right_on='SK_ID_CURR').sort_values('_P')
    
    feat_val_per_cust_cur = feat_val_per_cust_cur.T
    feat_val_per_cust = feat_val_per_cust_cur.merge(feat_val_per_cust, left_index=True, right_index=True, how = 'right')
    feat_val_per_cust.reset_index(inplace=True)    
    tab_labels = ['Variable', 'Courant', 'Réf. 1', 'Réf. 2', 'Réf. 3', 'Réf. 4', 'Importance']
    
    feat_imp.columns = ['index','importance']
    # On fusionne les variables importantes avec les variables du client
    feat_val_per_cust_2_2 = feat_val_per_cust.merge(feat_imp,
                                                    left_on='index',
                                                    right_on='index',
                                                    how='left')    
    feat_val_per_cust_2_2.columns = tab_labels

     
    filtering_expressions = filter.split(' && ')
    
    dff2 = feat_val_per_cust_2_2
    
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            dff2 = dff2.loc[getattr(dff2[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff2 = dff2.loc[dff2[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            dff2 = dff2.loc[dff2[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff2 = dff2.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

    return dff2.iloc[ 
        page_current*page_size: (page_current + 1)*page_size
    ].to_dict('records')



if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:




