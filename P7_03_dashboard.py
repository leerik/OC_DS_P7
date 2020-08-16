#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# OPENCLASSROOMS
# Parcours Data Scientist
# Projet 7: Implémentez un modèle de scoring
# Etudiant: Eric Wendling
# Mentor: Julien Heiduk
# Date: 16/08/2020

#################################
### Import des modules Python ###
#################################

import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

import os
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_table import DataTable

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import joblib
from joblib import dump, load

# -*- coding: utf-8 -*-

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

######################
### Initialisation ###
######################

colors = {'background_1': '#FFFFFF', # blanc
          'text_1': '#000000', # noir
          'background_2': '#C0C0C0', # gris clair
          'text_2': '#FF0000'} # rouge

path_features = '../features_app/'
path_results = '../results_app/'

##########################
### Import des données ###
##########################

# Jeu de données d'entraînement avec étiquettes
feature_names = joblib.load(path_features+'feature_names.joblib')

# Jeu de données de test avec étiquettes
test_features = joblib.load(path_features+'test_features.joblib')
test_ids_ = joblib.load(path_features+'test_ids.joblib')
test_pred = joblib.load(path_features+'test_pred.joblib')
test_labels = joblib.load(path_features+'test_labels.joblib')

# Jeu de données de test sans étiquette
test_sub_features = joblib.load(path_features+'test_sub_features.joblib')
test_sub_ids_ = joblib.load(path_features+'test_sub_ids.joblib')
test_sub_pred = joblib.load(path_features+'test_sub_pred.joblib')

# Importance des variables
feat_imp = joblib.load(path_results+'feat_imp.joblib')

##############################
### Traitement des données ###
##############################

# Conversion de la variable 'SK_ID_CURR' au format texte (string)
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

# Création d'un individu 'défaut' (pour l'affichage initial des graphes)
test_features_with_id_default = test_features_with_id.iloc[0,:].copy()
test_features_with_id_default = pd.DataFrame(test_features_with_id_default).T
test_features_with_id_default.iloc[0,1:] = 0
test_features_with_id_default.iloc[0,0] = '0_'

# Ajout de l'individu 'defaut' aux jeux de test
test_features_with_id = pd.concat([test_features_with_id,test_features_with_id_default])
test_sub_features_with_id = pd.concat([test_sub_features_with_id,test_features_with_id_default])

# Valeur des variables pour un individu (demande/dossier de crédit)
# Initialisation de la requête
feat_val_per_cust = test_sub_features_with_id[test_sub_features_with_id['SK_ID_CURR']=='0_'].T
feat_val_per_cust.columns = ['value']
feat_val_per_cust.reset_index(inplace=True)

# On fusionne les variables importantes avec les variables du dossier de crédit
feat_val_per_cust_2 = feat_val_per_cust.merge(feat_imp,
                                              left_on='index', right_on='feature').sort_values('importance',ascending=False)
feat_val_per_cust_2 = feat_val_per_cust_2[['feature','value','importance']]
feat_val_per_cust_2 = round(feat_val_per_cust_2,2)

#####################
### Métriques ROC ###
#####################

[fpr, tpr, thr] = metrics.roc_curve(test_labels, test_pred)
df_fpr = pd.DataFrame(fpr)
df_tpr = pd.DataFrame(tpr)

df_roc = df_fpr.merge(df_tpr, left_index=True, right_index=True)
df_roc.columns = ['fpr','tpr']

auc_val = round(metrics.auc(fpr, tpr),3)

metric_ = 'GainNorm'
solvability_threshold_g_norm = joblib.load(path_results+'thr_g_norm.joblib')
idx = np.max(np.where(thr > solvability_threshold_g_norm))

# Gain normalisé (dataframe)
df_g_norm = joblib.load(path_results+'df_g_norm.joblib')

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
g_norm_opt = df_g_norm['g_norm'].loc[id_]

###############
### Scoring ###
###############

# Le dataframe 'scoring' contient les valeurs de prédiction
scoring_default = pd.DataFrame({'SK_ID_CURR':'0_',
                                '_N':0.5,
                                '_P':0.5,
                                '_Pred':0},
                               index = [0])

# Scoring test
test_pred_bin = (test_pred > thr[idx])
test_pred_bin = np.array(test_pred_bin > 0) * 1
test_pred_bin = pd.DataFrame(test_pred_bin)

scoring_test = df_test_ids.reset_index(drop=True).merge(pd.DataFrame(test_pred), left_index=True, right_index=True)
scoring_test = scoring_test.merge(test_pred_bin, left_index=True, right_index=True)
scoring_test.columns = ['SK_ID_CURR','_P','_Pred']

scoring_test['_N'] = 1 - scoring_test['_P']
scoring_test = scoring_test[['SK_ID_CURR','_N','_P','_Pred']]
scoring_test = pd.concat([scoring_test,scoring_default])

# Scoring test - Mise à jour des listes de valeurs de l'identifiant 'SK_ID_CURR' (dropdownlist)
scoring_test = scoring_test.sort_values('_P')
list_test_credit_ids = scoring_test[(scoring_test['_P'] >= 0) & (scoring_test['_P'] <= 1)]
list_test_credit_ids = list_test_credit_ids.merge(df_test_ids_, left_on = 'SK_ID_CURR', right_on = 'SK_ID_CURR')
list_test_credit_ids = list_test_credit_ids[['SK_ID_CURR_num']]

list_test_credit_ids.loc[list_test_credit_ids.shape[0]+1,'SK_ID_CURR_num'] = 0
list_test_credit_ids['SK_ID_CURR_num']=list_test_credit_ids['SK_ID_CURR_num'].astype('int')

list_test_credit_ids = list_test_credit_ids['SK_ID_CURR_num']

# Scoring test_sub
test_sub_pred_bin = (test_sub_pred > thr[idx])
test_sub_pred_bin = np.array(test_sub_pred_bin > 0) * 1
test_sub_pred_bin = pd.DataFrame(test_sub_pred_bin)

scoring_sub = df_test_sub_ids.reset_index(drop=True).merge(pd.DataFrame(test_sub_pred), left_index=True, right_index=True)
scoring_sub = scoring_sub.merge(test_sub_pred_bin, left_index=True, right_index=True)
scoring_sub.columns = ['SK_ID_CURR','_P','_Pred']

scoring_sub['_N'] = 1 - scoring_sub['_P']
scoring_sub = scoring_sub[['SK_ID_CURR','_N','_P','_Pred']]
scoring_sub = pd.concat([scoring_sub,scoring_default])

# Scoring test_sub - Mise à jour des listes de valeurs de l'identifiant 'SK_ID_CURR' (dropdownlist)
scoring_sub = scoring_sub.sort_values('_P')
list_test_sub_credit_ids = scoring_sub[(scoring_sub['_P'] >= 0) & (scoring_sub['_P'] <= 1)]
list_test_sub_credit_ids = list_test_sub_credit_ids.merge(df_test_sub_ids_, left_on = 'SK_ID_CURR', right_on = 'SK_ID_CURR')
list_test_sub_credit_ids = list_test_sub_credit_ids[['SK_ID_CURR_num']]

list_test_sub_credit_ids.loc[list_test_sub_credit_ids.shape[0]+1,'SK_ID_CURR_num'] = 0
list_test_sub_credit_ids['SK_ID_CURR_num']=list_test_sub_credit_ids['SK_ID_CURR_num'].astype('int')

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

########################
### Fonction globale ###
########################

def metrics(recall, x, data_to_pred):
    
    idx = np.max(np.where(thr > recall))

    if recall < 0.5:
        scoring_default.loc[0,'_Pred'] = 1
    elif recall >= 0.5:
        scoring_default.loc[0,'_Pred'] = 0
    
    # test
    test_pred_bin = (test_pred > thr[idx])
    test_pred_bin = np.array(test_pred_bin > 0) * 1
    test_pred_bin = pd.DataFrame(test_pred_bin)

    scoring = df_test_ids.reset_index(drop=True).merge(pd.DataFrame(test_pred), left_index=True, right_index=True)
    scoring = scoring.merge(test_pred_bin, left_index=True, right_index=True)
    scoring.columns = ['SK_ID_CURR','_P','_Pred']

    scoring['_N'] = 1 - scoring['_P']
    scoring = scoring[['SK_ID_CURR','_N','_P','_Pred']]   
    scoring = pd.concat([scoring,scoring_default])
    
    # Prédiction (0 ou 1)
    res_sco_test = scoring[scoring['SK_ID_CURR']==x]['_Pred'].values.tolist() # OPTION_SUB

    # test_sub
    test_sub_pred_bin = (test_sub_pred > thr[idx])
    test_sub_pred_bin = np.array(test_sub_pred_bin > 0) * 1
    test_sub_pred_bin = pd.DataFrame(test_sub_pred_bin)

    scoring_sub = df_test_sub_ids.reset_index(drop=True).merge(pd.DataFrame(test_sub_pred), left_index=True, right_index=True)
    scoring_sub = scoring_sub.merge(test_sub_pred_bin, left_index=True, right_index=True)
    scoring_sub.columns = ['SK_ID_CURR','_P','_Pred']

    scoring_sub['_N'] = 1 - scoring_sub['_P']
    scoring_sub = scoring_sub[['SK_ID_CURR','_N','_P','_Pred']]
    scoring_sub = pd.concat([scoring_sub,scoring_default])
    
    # Prédiction (0 ou 1)    
    res_sco_sub = scoring_sub[scoring_sub['SK_ID_CURR']==x]['_Pred'].values.tolist() # OPTION_SUB
    
    res_sco = res_sco_sub
    
    if data_to_pred == "Subm":
        res_sco = res_sco_sub
    
    elif data_to_pred == "Test":
        res_sco = res_sco_test        

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
    g_norm_ch = round(g_norm_ch,2)

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
    
    #############
    ### POLAR ###
    #############

    df = pd.DataFrame(dict(r=[sensibilite, specificite, precision, f_mesure],
                           theta=['SE','SP','PR','F1']))
    
    fig_pol = px.line_polar(df, r='r', theta='theta', line_close=True)
    
    fig_pol.update_layout(plot_bgcolor=colors['background_1'],
                          paper_bgcolor='lightgrey',
                          font_color=colors['text_1'],
                          font_size=10,
                          margin=dict(l=20, r=20, t=20, b=20),
                          autosize=True,
                          #width=600,
                          height=180)
    
    ###########
    ### PIE ###
    ###########
    
    if data_to_pred == "Subm":
        scoring_pie = scoring_sub
    
    elif data_to_pred == "Test":
        scoring_pie = scoring       
        
    pie_default = 0
    
    if pie_default == 0:
        _0_count = scoring_pie.shape[0] - scoring_pie['_Pred'].sum()
        _1_count = scoring_pie['_Pred'].sum()
    
    elif pie_default == 1:
        _0_count = 71.8
        _1_count = 28.2

    df_pie = pd.DataFrame({
    'Classe': ['Solvables','Non Solvables'],
    'Nombre': [_0_count,_1_count]}, index=[0,1])
    
    fig_pie = px.pie(df_pie, values='Nombre', names='Classe',
                     color_discrete_sequence=['rgba(0, 153, 51, 0.8)', 'rgba(255, 51, 0, 0.8)'])
    
    fig_pie.update(layout_showlegend=False)
    
    fig_pie.update_traces(hoverinfo='label',
                          textinfo='percent',
                          textfont_size=11,
                          marker=dict(line=dict(color='lightgrey', width=3))
                         )

    fig_pie.update_layout(plot_bgcolor=colors['background_1'],
                          paper_bgcolor='lightgrey',
                          font_color=colors['text_1'],
                          font_size=11,
                          margin=dict(l=20, r=20, t=10, b=10),
                          autosize=True,
                          #width=600,
                          height=150)
   
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

    fig.update_layout(margin=dict(t=10, l=20),
                      autosize=True,
                      font_size=11,
                      #width=400,
                      height=170,
                      paper_bgcolor='lightgrey')

    fig['data'][0]['showscale'] = True

    return auc_val,round(thr[idx],2),round(tpr[idx],2),round(1-fpr[idx],2),res_sco,g_norm_ch,precision,f_mesure,fig,fig_pol,fig_pie

########################
### Application Dash ###
########################

# Tableau: Importance des variables
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# Graduation des Sliders
marks = np.arange(0,1.1,0.1)
df_marks = pd.DataFrame(marks)
df_marks.columns = ['marks']
df_marks = round(df_marks,2)

marks_2 = np.arange(0,1,0.02)
df_marks_2 = pd.DataFrame(marks_2)
df_marks_2.columns = ['marks']
df_marks_2 = round(df_marks_2,2)

# Fonction de création d'une bannière
def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            
            html.Div(className='row tabs_div', # ligne 2
                 children=[
                     html.Div(className='two columns',
                              id="banner-logo",
                              children=[
                                  html.Img(id="logo", src="http://placeimg.com/625/225/animals", height='50', width='50'),
                         ],
                     ),
                     
                     html.Div(className='ten columns',
                              id="banner-text",
                              children=[
                                  html.H1('Credit Scoring'),
                         ]
                     )
                 ]
                    )
        ]
    )

# Layout

app.layout = html.Div( # conteneur principal
    children=[
        html.Div(className='row tabs_div', # L1 (ligne 1)
                 children=[
                     
                     html.Div(), # L1C0 (ligne 1, colonne 0)
                     
                     html.Div(className='three columns', # L1C1 (ligne 1, colonne 1)
                              children=[
                                  html.Br(),
                                  build_banner(),
                                  html.H6("Numéro de dossier de crédit"),
                                  
                                  html.Div(className='row tabs_div', # L1C1-L1
                                           children=[
                                               html.Div(className='seven columns', # L1C1-L1C1
                                                        children=[
                                                            dcc.Dropdown(
                                                                id='my-input',
                                                                # Les options sont définies par la fonction set_input_options
                                                                value= '0' # Valeur par défaut du numéro de dossier
                                                            ),
                                                        ]
                                                       ),
                                               html.Div(className='five columns', # L1C1-L1C2
                                                        children=[
                                                            dcc.RadioItems(
                                                                id='data-to-pred',
                                                                options=[{'label': i, 'value': i} for i in ['Test', 'Subm']],
                                                                value='Test',
                                                                labelStyle={'display': 'inline-block'}
                                                            ),
                                                        ]
                                                       )
                                           ]
                                          ), # fin L1C1-L1
                                  
                                  html.Div(className='row tabs_div', # L1C1-L2
                                           children=[
                                               html.Div(className='seven columns', # L1C1-L2C1
                                                        children=[
                                                            html.Table([
                                                                html.Tr([html.Td(['Seuil']), html.Td(id='seuil')])
                                                            ])
                                                        ]
                                                       ),
                                               html.Div(className='five columns', # L1C1-L2C2
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
                     
                     html.Div(className='two columns', # L1C2 (ligne 1, colonne 2)
                              children=[
                                  html.Br(),                                  
                                  dcc.Graph(id='polar'),
                                  html.Br(),
                                  html.Br(),
                                  dcc.Graph(id='pie'),
                              ]
                             ), # fin L1C2
                     
                     html.Div(className='three columns', # L1C3 (ligne 1, colonne 3)
                              children=[
                                  html.Br(),
                                  
                                  html.Div(className='row tabs_div', # L1C3-L1
                                           children=[
                                               html.Div(className='six columns', # L1C3-L1C1
                                                        children=[
                                                            html.Table([
                                                                html.Tr([html.Td(['AUC']), html.Td(id='auc')]),
                                                                html.Tr([html.Td(['Gain']), html.Td(id='gain')]),
                                                                html.Tr([html.Td(['F-Mesure']), html.Td(id='f_mesure')])
                                                            ])
                                                        ]
                                                       ),
                                               html.Div(className='five columns', # L1C3-L1C2
                                                        children=[
                                                            html.Table([
                                                                
                                                                html.Tr([html.Td(['Sensibilité']), html.Td(id='sensibilite')]),
                                                                html.Tr([html.Td(['Spécificité']), html.Td(id='specificite')]),
                                                                html.Tr([html.Td(['Précision']), html.Td(id='precision')]),
                                                                
                                                            ])
                                                        ]
                                                       )
                                           ]
                                          ), # fin L1C3-L1                                 
                                  
                                  html.Br(),
                                  dcc.Graph(id='heatmap'),
                              ]
                             ), # fin L1C3
                     
                     html.Div(className='three columns', # L1C4 (ligne 1, colonne 4)
                              children=[
                                  html.H6(children='Simulation seuil',style={'textAlign': 'left','color': colors['text_1']}),
                                  dcc.Graph(id='simulation-roc'),
                                  html.Br(),
                                  dcc.Slider(
                                      id='slider-threeshold', # seuil
                                      min=0,
                                      max=1.1,
                                      value=solvability_threshold_g_norm,
                                      marks={str(mark): str(mark) for mark in df_marks['marks']},
                                      step=None,
                                  )
                              ]
                             ) # fin L1C4
                 ],
#                  style={
#                      'border':'1px white solid',
#                      'background-color': 'lightgrey'}
                ), # fin L1

        
        html.Div(className='row', # L2 (ligne 2)
                 children=[
                     
                     html.Div(), # L2C0 (ligne 2, colonne 0)
                     
                     html.Div(className='three columns', # L2C1 (ligne 2, colonne 1)
                              children=[
                                  html.Br(),
                                  dcc.Graph(id='fig-scoring'),
                              ]
                             ), # fin L2C1
                     
                     html.Div(className='two columns', # L2C2 (ligne 2, colonne 2)
                              children=[
                                  html.H4(children='',style={'textAlign': 'left','color': colors['text_1']}),
                                  html.Div([
                                      ###
                                  ]),
                              ]
                             ), # fin L2C2
                     
                     html.Div(className='three columns', # L2C3 (ligne 2, colonne 3)
                              children=[
                                  dcc.Graph(id='heatmap-2'),
                                  dcc.Graph(id='heatmap-3')
                              ],
                             ), # fin L2C3
                     
                     html.Div(className='three columns', # L2C4 (ligne 2, colonne 4)
                              children=[
                                  html.H6(children='Simulation gain',style={'textAlign': 'left','color': colors['text_1']}),
                                  dcc.Graph(id='simulation-gain'),
                                  html.Br(),
                                  dcc.Slider(
                                      id='slider-gain',
                                      min=df_g_norm['g_norm'].min(),
                                      max=df_g_norm['g_norm'].max(),
                                      value=0.58, # Valeur par défaut du gain
                                      marks={str(mark): str(mark) for mark in df_marks_2['marks']},
                                      step=None,
                                      included=True
                                  )
                              ]
                             ) # fin L1C4
                 ]
                ), # fin L3
        
        
        html.Div(className='row', # L4 (ligne 4)
                 children=[
                     
                     html.Div(), # L4C0 (ligne 4, colonne 0)
                     
                     html.Div(className='five columns', # L4C1 (ligne 4, colonne 1)
                              children=[
                                  html.H6(children='Variables',style={'textAlign': 'left','color': colors['text_1']}),
                                  dash_table.DataTable(
                                      id='table-paging-with-graph',
                                      columns=[
                                          {"name": i, "id": i} for i in (feat_val_per_cust_2.columns)
                                      ],
                                      page_current=0,
                                      page_size=10, # Nombre de lignes par page
                                      page_action='custom',
                                      filter_action='custom',
                                      filter_query='',
                                      sort_action='custom',
                                      sort_mode='multi',
                                      sort_by=[]
                                  ),
#                                   html.Br(),
#                                   html.Br(),
#                                   html.Div(
#                                       id='table-paging-with-graph-container',
#                                       className="five columns"
#                                   )
                              ]
                             ), # fin L4C1
                     
                     html.Div(className='two columns', # L4C2 (ligne 4, colonne 2)
                              children=[
                                  html.H6(children='Titre L4C2',style={'textAlign': 'left','color': colors['text_1']}),
                                  html.Div([
                                      ###
                                  ]),
                              ]
                             ), # fin L4C2
                     
                     html.Div(className='two columns', # L4C3
                              children=[
                                  html.H6(children='Titre L4C3',style={'textAlign': 'left','color': colors['text_1']}),
                                  html.Div([
                                      ####
                                  ]),
                              ]
                             ), # fin L4C3
                     
                     html.Div(className='two columns', # L4C4
                              children=[
                                  html.H6(children='Titre L4C4',style={'textAlign': 'left','color': colors['text_1']}),
                                  html.Div([
                                      ###
                                  ]),
                              ]
                             ), # fin L4C4
                 ]
                ) # fin L4
    
    ],style={
#         'border':'1px grey solid',
        'background-color': 'lightgrey'}
    
) # fin layout

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


@app.callback(
    Output('my-input', 'options'),
    [Input('data-to-pred', 'value')])

def set_input_options(data_to_pred):
    if data_to_pred == 'Subm':
        return [{'label': i, 'value': i} for i in list_test_sub_credit_ids]
    elif data_to_pred == 'Test':
        return [{'label': i, 'value': i} for i in list_test_credit_ids]


@app.callback(
    Output('simulation-roc', 'figure'),
    [Input('slider-threeshold', 'value')])

def update_roc(thr_ch):

    idx = np.max(np.where(thr > thr_ch))
#     print ("Seuil : %.2f" % thr[idx])

    x0 = df_roc['fpr']
    x1 = [0, fpr[idx]]
    x2 = [fpr[idx], fpr[idx]]

    fig_roc = go.Figure()

    fig_roc.add_trace(go.Scatter(
        x=x1,
        y=[tpr[idx],tpr[idx]],
        name = 'Sensibilité',
        connectgaps=True
    ))

    fig_roc.add_trace(go.Scatter(
        x=x2,
        y=[tpr[idx],0],
        name='1 - Spécificité',
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

    fig_roc.update_layout(
        
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
        margin=dict(l=20, r=20, t=40, b=20),
        autosize=True,
        height=230
    )
    
    return fig_roc


@app.callback(
    Output('simulation-gain', 'figure'),
    [Input('slider-gain', 'value')])

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
#         marker_color='rgba(0, 153, 51, 0.8)'
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
        height=190
    )

    return fig_gain


@app.callback(
    [Output('fig-scoring', 'figure'),
     Output('fig-gauge', 'figure')],
    [Input(component_id='my-input', component_property='value'),
     Input('slider-threeshold', 'value'),
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
    elif input_value_str == worst_record_['SK_ID_CURR'].iloc[0]:
        filtered_scoring = pd.concat([filtered_scoring_current,best_record_])
    else:
        filtered_scoring = pd.concat([filtered_scoring_current,best_worst_records_])
        filtered_scoring = filtered_scoring.sort_values('_P', ascending=False)
    
    credit = filtered_scoring['SK_ID_CURR']
    yp = round(filtered_scoring['_P'],2) # Probabilité non-solvabilité (positif: 0)
    yn = round(filtered_scoring['_N'],2) # Probabilité solvabilité (negatif: 1)
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
                color='rgba(255, 51, 0, 0.8)',#ici
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
    
    if input_value_str == best_record['SK_ID_CURR'].iloc[0]:
        proba_ = pd.DataFrame(yp).iloc[0,0]
        proba_worst_ = pd.DataFrame(yp).iloc[1,0]
    
    elif input_value_str == worst_record['SK_ID_CURR'].iloc[0]:
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
            'axis': {'range': [None, proba_worst_], 'tickwidth': 1, 'tickcolor': "darkblue"},
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
                'value': 0.5}}))

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
     Output('heatmap', 'figure')],
    [Input('slider-threeshold', 'value'),
     Input(component_id='my-input', component_property='value'),
     Input('data-to-pred', 'value')])

def metrics_heatmap_1(thr_ch, x, data_to_pred):
    
    x = str(x) + '_'
    
    a,b,c,d,e,f,g,h,i,j,k = metrics(thr_ch, x, data_to_pred)

    return a,b,c,d,e,f,g,h,i
    

@app.callback(
    Output('heatmap-2', 'figure'),
    [Input('slider-gain', 'value'),
     Input(component_id='my-input', component_property='value'),
     Input('data-to-pred', 'value')])

def heatmap_2(gain_ch, x, data_to_pred):
    
    x = str(x) + '_'
    
    df_g_norm_filt = df_g_norm[df_g_norm['g_norm'] >= gain_ch]
    seuil_min = df_g_norm_filt[['g_norm','seuil']].sort_values('seuil').iloc[0]['seuil']
    
    a,b,c,d,e,f,g,h,i,j,k = metrics(seuil_min, x, data_to_pred)
    
    return i


@app.callback(
    Output('heatmap-3', 'figure'),
    [Input('slider-gain', 'value'),
     Input(component_id='my-input', component_property='value'),
     Input('data-to-pred', 'value')])

def heatmap_3(gain_ch, x, data_to_pred):
    
    x = str(x) + '_'
    
    df_g_norm_filt = df_g_norm[df_g_norm['g_norm'] >= gain_ch]
    seuil_max = df_g_norm_filt[['g_norm','seuil']].sort_values('seuil', ascending=False).iloc[0]['seuil']
    
    a,b,c,d,e,f,g,h,i,j,k = metrics(seuil_max, x, data_to_pred)
    
    return i


@app.callback(
    Output('polar', 'figure'),
    [Input('slider-threeshold', 'value'),
     Input(component_id='my-input', component_property='value'),
     Input('data-to-pred', 'value')])

def fig_polar(thr_ch, x, data_to_pred):
    
    x = str(x) + '_'
 
    a,b,c,d,e,f,g,h,i,j,k = metrics(thr_ch, x, data_to_pred)
    
    return j


@app.callback(
    Output('pie', 'figure'),
    [Input('slider-threeshold', 'value'),
     Input(component_id='my-input', component_property='value'),
     Input('data-to-pred', 'value')])

def fig_pie(thr_ch, x, data_to_pred):
    
    x = str(x) + '_'
 
    a,b,c,d,e,f,g,h,i,j,k = metrics(thr_ch, x, data_to_pred)
    
    return k


@app.callback(
    Output('table-paging-with-graph', "data"),
    [Input('table-paging-with-graph', "page_current"),
     Input('table-paging-with-graph', "page_size"),
     Input('table-paging-with-graph', "sort_by"),
     Input('table-paging-with-graph', "filter_query"),
     Input(component_id='my-input', component_property='value'),
     Input('data-to-pred', 'value')])

def features_importance_tab(page_current, page_size, sort_by, filter, x, data_to_pred):
    
    x = str(x) + '_'
    
    if data_to_pred == "Subm":
        test_features_with_id_ = test_sub_features_with_id # OPTION_SUB
    
    elif data_to_pred == "Test":
        test_features_with_id_ = test_features_with_id
    
    feat_val_per_cust = test_features_with_id_[test_features_with_id_['SK_ID_CURR']==x].T
    feat_val_per_cust.columns = ['value']
    feat_val_per_cust.reset_index(inplace=True)

    # On fusionne les variables importantes avec les variables du client
    feat_val_per_cust_2 = feat_val_per_cust.merge(feat_imp,
                                                  left_on='index', right_on='feature').sort_values('importance',ascending=False)
    feat_val_per_cust_2 = feat_val_per_cust_2[['feature','value','importance']]
    feat_val_per_cust_2 = round(feat_val_per_cust_2,3)

     
    filtering_expressions = filter.split(' && ')
    dff = feat_val_per_cust_2
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
    Output('table-paging-with-graph-container', "children"),
    [Input('table-paging-with-graph', "data")])

def features_importance_fig(rows):
    dff = pd.DataFrame(rows)
    return html.Div(
        [
            dcc.Graph(
                id=column,
                figure={
                    "data": [
                        {
                            "x": dff["feature"],
                            "y": dff[column] if column in dff else [],
                            "type": "bar",
                            "marker": {"color": "#0074D9"},
                        }
                    ],
                    "layout": {
                        "xaxis": {"automargin": True},
                        "yaxis": {"automargin": True},
                        "height": 250,
                        "margin": {"t": 10, "l": 10, "r": 10},
                    },
                },
            )
            for column in ["importance"]
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=False)

