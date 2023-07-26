# Hide & Seek deep RL
## Tables des matières
* [Informations Générales](#general-info)
* [Architecture du Projet](#architecture)
* [Apprentissage & Résultats](#results)
* [Setup](#setup)

## Informations Générales

Ce projet présente un simple jeu de hide & seek où un agent (player) apprend à se cacher d'un ennemi statique sur une map quadrillée de taille 12*12.

Pour ce faire, nous utilisons l'implémentation de l'approche de Deep Q-Learning DQN basée sur le framework stable baselines 3 (SB3).

Nous définissons les notions principales, observations, actions et rewards de la sorte:

### Notions Principales

Notre jeu se base sur une map quadrillé, une notion de vision des différentes unités (agent et ennemi) ainsi qu'une notion de "point d'intérêt" pour l'agent.

Plus précisemment, nous considérons une map de jeu comme une matrice 12*12 composées de 7 valeurs différentes:
* 1 = case vide
* 2 = block infranchissable
* 3 = hors des limites de la map (utilisé dans certains cas)
* 4 = agent, joueur essayent de se cacher
* 5 = ennemi
* 6 = case vide mais dans la vision de l'ennemi
* 7 = point d'intérêt pour l'agent

La notion de point d'intêret représente la case qui semble être la plus intéressante pour un agent à un instant t.
Brièvement, on calcule un point d'intérêt de la sorte:
*  Si l'agent est dans le champ de vision de l'ennemi, le point d'intérêt de l'agent est alors la case la plus proche de lui où il pense que l'ennemi ne le verra pas
*  Sinon
    * Si l'agent voit actuellement une case adjacente à deux blocks, il considère qu'elle est adéquate pour se cacher et la définit donc comme son point d'intérêt.
        * Sinon
            * Si l'agent estime qu'il n'a pas encore assez exploré la carte, il définit comme point d'intérêt la case la plus proche qu'il n'a pas encore vu.
                * Sinon
                  * L'agent essaye de se rapprocher d'un block, afin d'être mieux caché. Il définit comme point d'intérêt le block le plus proche de lui.
                 
### Observations

Nous expérimentons avec 4 type d'observations, représentées par les 4 fichiers dans le dossier "envs/custom_envs"

#### CoordFieldVisionEnv (CFV)

Dans cette approche, les observations sont composés de deux positions (paires de coodonnées) et d'une distance.
Plus précisément: des coordonnées de l'agent, des coordonnées de la case qu'il doit emprunter pour se rapprocher de son point d'intérêt et de la distance entre les deux.  
Exemple:  
![Image](/readme_imgs/CFV_ex1.png)

Si X représente l'agent, @ représente un block et $ le point d'intérêt: le vecteur d'observation sera (1, 1, SEP, 1, 2, SEP, 2) où SEP est une valeur de séparation (= -1)
        
#### FullDirectionOnFieldEnv (FDF)

Dans cette approche, les observations sont composés de cinq valeurs, représentant l' action la plus probable que l'agent doit effectuer pour serapprocher de son point d'intêret.
Plus précisément: des valeurs binaires représentant les actions possibles par l'agent (gauche, droite, haut, bas, arrêt).  
Exemple:  
        ( .  .  .  . )  
        ( .  X  .  $ )  
        ( @  @  .  . )  
        ( .  .  .  . )  
        
Si X représente l'agent, @ représente un block et $ le point d'intérêt: le vecteur d'observation sera (0, 1, 0, 0, 0)

#### SpiralFieldVisionEnv (SFV)

Dans cette approche, les observations sont composés d'un vecteur de 9*9 valeurs représentant les "alentours" de l'agent, ordonné en forme de spirale avec pour origine l'agent.
Un example d'ordre de spirale serait:  
        9 10 11 12  
        8  1  2 13  
        7  X  3 14  
        6  5  4 ...  
  
Avec l'exemple précédent:  
        ( .  .  .  . )  
        ( .  X  .  $ )  
        ( @  @  .  . )  
        ( .  .  .  . )  

Si X représente l'agent, @ représente un block et $ le point d'intérêt, et en considérant les représentations numériques des cases ( . = 1 , @ = 2, $ = 7):
le vecteur d'observation sera (1, 1, 1, 1, 2, 2, 1, 1, outside_value, outside_value, outside_value, outside_value, 1, 7, etc.)

#### FullFieldVisionEnv (FFV)

Dans cette approche, les observations sont une image RGB 32*32 représentant la map, le positionnement des joueurs, la vision de l'ennemi (perçue par l'agent) et le point d'intêret.
Pour ce faire, nous "triplons" le contenu de la matrice représentant la map et créons 3 channels représentant les channels RGB afin d'obtenir une image (les cases ont des valeurs différentes pour chaque channel).
  
Avec l'exemple précédent:  
        ( .  .  .  . )  
        ( .  X  .  $ )  
        ( @  @  .  . )  
        ( .  .  .  . )  
  
On obtiendrait une image "triplée":  
        ( .  .  .  .  .  .  .  .  .  .  .  . )  
        ( .  .  .  .  .  .  .  .  .  .  .  . )  
        ( .  .  .  .  .  .  .  .  .  .  .  . )  
        ( .  .  .  X  X  X  .  .  .  $  $  $ )  
        ( .  .  .  X  X  X  .  .  .  $  $  $ )  
        ( .  .  .  X  X  X  .  .  .  $  $  $ )  
        ( @  @  @  @  @  @  .  .  .  .  .  . )  
        ( @  @  @  @  @  @  .  .  .  .  .  . )  
        ( @  @  @  @  @  @  .  .  .  .  .  . )  
        ( .  .  .  .  .  .  .  .  .  .  .  . )  
        ( .  .  .  .  .  .  .  .  .  .  .  . )  
        ( .  .  .  .  .  .  .  .  .  .  .  . )  
laquelle on peut transformer en trois channels différents.

Nous proposons deux variantes: 
* Soit l'image est statique, et représente la map dans son intégralité (mode_vision="static")
* Soit l'image est centrée sur l'agent (mode_vision="dynamic")

Dans le cas de la vision dynamique, du padding est ajoutée autour de la map afin de pouvoir obtenir une image 32*32 même si l'agent n'est pas centré (si il est au bords de la mappar exemple).

### Actions

Les actions possibles pour l'agent sont au nombre de 5:
* Mouvement vers la case à sa gauche
* Mouvement vers la case à sa droite
* Mouvement vers la case au dessus de lui
* Mouvement vers la case en dessous de lui
* Arrêt du jeu

### Rewards

Les rewards (à chaque 'step') sont les suivants:
* Si l'agent effectue un déplacement interdit (contre un block, l'ennemi ou essaye de sortir des limites de la map): il est pénalisé
* Si l'agent n'est pas dans le champ de vision de l'ennemi, il est récompensé
* Si l'agent se positionne contre plus d'un seul block, il est considéré comme "bien caché" et est récompensé
* Si l'agent se place de manière à ce qu'il y a plusieurs blocks entre lui et l'ennemi, il est récompensé
* Si l'agent s'éloigne deson point d'intêret, il est pénalisé. Sinon, il est récompensé.

## Architecture du Projet
L'architecture du projet est présentée ci-dessous.

L'environnement abstrait d'un jeu hide & seek est décrit dans le dossier "envs" à l'aide des fichiers "hideSeekEnv.py", "maps.py" et "unit.py".
Cet environement abstrait est implémenté de plusieurs manières dans le dossier "envs/custom_envs"

Le dossier "data" contient les informations relatives aux données externes (notamment les maps, définissables au format textuel)

Le dossier "misc" contient des fonctions génériques

Le dossier "scripts" contient les scripts exécutables

Le dossier "learned_models" contient des modèles déjà appris (un pour chaque type d'environnement)

├── envs    - Dossier contenant les informations concernant les différents environnements de jeu
│   ├── hideSeekEnv.py   - Fichier contenant la classe abstraite "HideSeekEnv" représentant un environnement générique de jeu hide & seek
│   ├── maps.py        - Fichier contenant la classe Maps représentant une map sur laquelle se déroule le jeu de hide & seek
│   ├── unit.py    - Fichier contenant la classe Unit représentant une unité (joueur/ennemi) dans une partie de hide & seek
│   └── custom_envs    - Dossier contenant les environnement implémentant la classe abstraite "HideSeekEnv"
|       ├── coordFieldVision.py   - Fichier contenant la classe représentant un environnement de type "CoordFieldVisionEnv"
│       ├── fullDirectionOnField.py   - Fichier contenant la classe représentant un environnement de type "FullDirectionOnFieldEnv"
│       ├── fullFieldVision.py   - Fichier contenant la classe représentant un environnement de type "FullFieldVisionEnv"
│       └── spiralFieldVision.py   - Fichier contenant la classe représentant un environnement de type "SpiralFieldVisionEnv"
│
│
├── data               - Dossier contenant les données externes nécessaires au jeu de hide & seek
│   ├── dataloader.py  - Fichier contenant une simple classe transformant des données externes en objets exploitables par les environnements
│   ├── opti_training.py - Fichier contenant des optimisations (données pré-calculées) pour rendre le training plus efficient
│   └── example_maps    - Dossier contenant les données externes (sous forme de fichiers textuels) concernant les différentes maps de jeu
│       └── map_v1
│       └── map_v2
│       └── map_v3
│       └── map_v4
│
│
├── misc             
│   └── utils.py   - Fichier contenant des fonctions "génériques" utilisées par les différents environements
│   
├──  scripts              - Dossier contenant les scripts exécutables
│    ├── eval.py    - Script permettant d'évaluer la performance d'un modèle
│    └── train.py    - Script permettant de lancer le training d'un modèle
│  
└──  learned_models           - Dossier contenant les modèles appris
     ├── model_CFV.zip  - Modèles appris exploitant l'environnement de type "CoordFieldVisionEnv"
     └── model_FDF.zip  - Modèles appris exploitant l'environnement de type "FullDirectionOnFieldEnv"
     └── model_FFV.zip  - Modèles appris exploitant l'environnement de type "FullFieldVisionEnv"
     └── model_SFV.zip  - Modèles appris exploitant l'environnement de type "SpiralFieldVisionEnv"

## Apprentissage & Résultats

Les modèles sont appris avec les paramètres par défaut de l'implémantation DQN de SB3, sauf le paramètre "exploration_fraction" qui est définit à 0.20 car cela permettait à l'agent une meilleure exploration des possibilités de mouvement dans notre cas.
Chaque modèle est appris pour 700 000 steps.

Dans le cas des approches CoordFieldVisionEnv, FullDirectionOnFieldEnv et SpiralFieldVisionEnv, "MlpPolicy" est utilisé en tant que policy.
Dans le cas de l'approche FullFieldVisionEnv, "CnnPolicy" est utilisé en tant que policy.

## Setup

Afin de faire fonctionner nos modèles, le package SB3 doit être installé:

!pip install "stable-baselines3[extra]>=2.0.0a4"

git clone https://github.com/lucassA/DeepRL.git
