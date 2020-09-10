#### OPENCLASSROOMS - Parcours Data Scientist  
Etudiant: Eric Wendling  
Mentor: Julien Heiduk  
Date: 09/09/2020

## Projet 7: Implémentez un modèle de scoring  

Le **Crédit Scoring** est un outil d'analyse de risque pour l'octroi de crédits. Basé sur des méthodes statistiques, il prend en compte de nombreuses informations relatives au demandeur de crédit pour évaluer le risque de non remboursement. Concrètement, l'outil affecte un score à une demande de crédit par analyse statistique sur une base de référence (dossiers échus dont on connaît l'issue).

Bien que très répandu dans les organismes de crédit, de nombreux aspects restent à améliorer. 

**Confiance**

>L'impartialité de l'outil est souvent citée comme vertu par l'industrie mais ne convainc pas nécessairement les demandeurs surtout lorsqu'ils se voient refuser leur demande de crédit.

**Transparence des informations**

>Le nouveau règlement européen sur la protection des données personnelles est entré en application le 25 mai 2018 (https://www.cnil.fr/fr/reglement-europeen-protection-donnees).
>
>L'Article 12 traite particulièrement de la "Transparence des informations et des communications et modalités de l'exercice des droits de la personne concernée".

**Performance**

>Quelle que soit la méthode utilisée pour établir les scores des demandes de crédit, les résultats peuvent être trompeurs. 
>
>Cela peut être le cas pour les dossiers "hors-normes" par exemple, qui rencontrent peu de cas similaires dans l'échantillon de traitement. La qualité des données de référence est donc très importante. Il s'agit de disposer d'une base la plus représentative possible des situations à traiter.
>
>Les techniques et méthodes utilisées pour construire le modèle ont bien entendu un impact important sur sa performance. Aucun modèle n'est parfait et l'on rencontrera toujours des cas de mauvaise prédiction. L'enjeu technique est de minimiser ces cas.

On se propose ici de réaliser un outil de **Crédit Scoring** basé sur des technologies de **Machine Learning**.

Le Machine Learning a la réputation d'être peu transparent et de se rapprocher d'une boîte noire. Cela ne favorise pas son acceptation eu égard aux deux premiers points cités plus haut. Néanmoins, d'énormes progrès ont été réalisés à ce jour et le domaine du Machine Learning offre des solutions très intéressantes pour améliorer la compréhension des processus de décision.

**Modélisation**

L'objectif est double:

+ Le modèle doit permettre de définir la ***probabilité de défaut de remboursement*** d'un crédit sur la base d'informations relatives au client.
+ Il doit également offrir un certain niveau de transparence concernant les données et leurs traitements en vue d'implémenter des méthodes d'***interprétabilité*** des variables.

>Nous allons modéliser l'outil en utilisant des technologies d'**apprentissage supervisé**.

**Application**

Au-delà des aspects techniques, la transparence de l'outil se caractérise également par les possibilités d'interaction avec ce dernier en vue de réaliser des analyses complémentaires sur la base des résultats proposés.

+ On veut par exemple pouvoir comparer 2 dossiers similaires dont les prédictions d'octroi de crédit sont différentes et visualiser les variables ayant influencé les décisions.
+ On peut également vouloir réaliser des simulations pour estimer à quel degré un dossier a été refusé et identifier les critères discriminants.

>Nous allons déployer le modèle via une application web en utilisant **Dash**.

## Plan

Le projet a été découpé en trois parties traitées respectivement dans trois notebooks Jupyter.

**Partie 1**  
https://github.com/leerik/OC_DS_P7/blob/master/P7_01_analyse.ipynb
>La première partie du projet consiste à réaliser l'analyse exploratoire des données et leur traitement en vue de construire le(s) dataframe(s) adapté(s) à la modélisation d'un outil de Machine Learning.

**Partie 2**  
Notebook P7_02_scoring

>La deuxième partie est consacrée à la modélisation d'un système d'apprentissage supervisé.

**Partie 3**  
Notebook P7_03_dashboard
>La troisième partie concerne le déploiement du modèle via le web.
