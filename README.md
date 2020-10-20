#### OPENCLASSROOMS - Parcours Data Scientist  
Etudiant: Eric Wendling  
Mentor: Julien Heiduk  
Date: 20/10/2020

## Projet 7: Implémentez un modèle de scoring  

Le **Crédit Scoring** est un outil d'analyse de risque pour l'octroi de crédits. Basé sur des méthodes statistiques, il prend en compte de nombreuses informations relatives au demandeur de crédit pour évaluer le risque de non remboursement. Concrètement, l'outil affecte un score à une demande de crédit par analyse statistique sur une base de référence (dossiers échus dont on connaît l'issue).

Bien que très répandu dans les organismes de crédit, de nombreux aspects restent à améliorer. 

**Confiance**

L'impartialité de l'outil est souvent citée comme vertu par l'industrie mais ne convainc pas nécessairement les demandeurs surtout lorsqu'ils se voient refuser leur demande de crédit.

**Transparence des informations**

Le nouveau règlement européen sur la protection des données personnelles est entré en application le 25 mai 2018 (https://www.cnil.fr/fr/reglement-europeen-protection-donnees). L'Article 12 traite particulièrement de la "Transparence des informations et des communications et modalités de l'exercice des droits de la personne concernée".

**Performances**

Quelle que soit la méthode utilisée pour établir les scores des demandes de crédit, les résultats peuvent être trompeurs. Cela peut être le cas pour les dossiers "hors-normes" par exemple, qui rencontrent peu de cas similaires dans l'échantillon de traitement.

>La qualité des données de référence est donc très importante. Il s'agit de disposer d'une base la plus représentative possible des situations à traiter.

Les techniques et méthodes utilisées pour construire le modèle ont bien entendu un impact important sur sa performance. Aucun modèle n'est parfait et l'on rencontrera toujours des cas de mauvaise prédiction.

>L'enjeu technique est de minimiser ces cas.

## Projet

On se propose ici de réaliser un outil de **Crédit Scoring** basé sur des technologies de **Machine Learning**. Le Machine Learning a la réputation d'être peu transparent et de se rapprocher d'une boîte noire. Cela ne favorise pas son acceptation eu égard aux deux premiers points cités plus haut.

>Néanmoins, d'énormes progrès ont été réalisés à ce jour et le domaine du Machine Learning offre des solutions très intéressantes pour améliorer la compréhension des processus de décision.

**Modélisation**

L'objectif est double:

+ Le modèle doit permettre de définir la ***probabilité de défaut de remboursement*** d'un crédit sur la base d'informations relatives au client.
+ Il doit également offrir un certain niveau de transparence concernant les données et leurs traitements en vue d'implémenter des méthodes d'***interprétabilité*** des variables.

>Nous avons modélisé l'outil en utilisant des technologies d'**apprentissage supervisé**.

Les principales étapes de la modélisation sont décrites dans la note méthodologique:

https://github.com/leerik/OC_DS_P7/blob/master/P7_04_note_methodologique.pdf

**Application**

Au-delà des aspects techniques, la transparence de l'outil se caractérise également par les possibilités d'interaction avec ce dernier en vue de réaliser des analyses complémentaires sur la base des résultats proposés.

+ On veut par exemple pouvoir comparer 2 dossiers similaires dont les prédictions d'octroi de crédit sont différentes et visualiser les variables ayant influencé les décisions.
+ On peut également vouloir réaliser des simulations pour estimer à quel degré un dossier a été refusé et identifier les critères discriminants.

>Nous avons déployé le modèle via une application web en utilisant **Dash**.

## Plan

Le projet a été découpé en trois parties traitées respectivement dans trois notebooks Jupyter.

**Partie 1**  

La première partie du projet consiste à réaliser l'analyse exploratoire des données et leur traitement en vue de construire le(s) dataframe(s) adapté(s) à la modélisation d'un outil de Machine Learning.

https://github.com/leerik/OC_DS_P7/blob/master/P7_01_analyse.ipynb

**Partie 2**  

La deuxième partie est consacrée à la modélisation d'un système d'apprentissage supervisé.

https://github.com/leerik/OC_DS_P7/blob/master/P7_02_scoring.ipynb

**Partie 3**  

La troisième partie concerne le développement de l'application web (Dash).

https://github.com/leerik/OC_DS_P7/blob/master/P7_03_dashboard.ipynb

## Déploiement

**Local**

Pour lancer l'application en local, il suffit d'exécuter le notebook P7_03_dashboard.ipynb.

**Serveur distant**

Le code de l'application est inscrit dans le script python suivant:

https://github.com/leerik/OC_DS_P7/blob/master/P7_03_dashboard.py

Le déploiement du script est réalisé via la plateforme Heroku. L'application peut être chargée avec le lien suivant:

https://credit-scoring-dashboard.herokuapp.com/

***Avertissements***

Pour des raisons de gestion des ressources serveur, Heroku ne garde pas en mémoire les applications qui ne sont pas en cours d'utilisation.
Ainsi, le premier chargement de l'application nécessite un certain laps de temps (en moyenne 20 secondes).

On notera également que les fonctionnalités d'inteprétabilité des variables (utilisant la technologie SHAP) ne sont disponibles qu'en mode local.








