# Hide & Seek deep RL
## Tables des matières
* [Informations Générales](#general-info)
* [Architecture du Projet](#architecture)
* [Expérimentations](#expérimentations)
* [Setup](#setup)

## Informations Générales

Ce projet présente un simple jeu de hide & seek où un agent (player) apprend à se cacher d'un ennemi statique sur une map quadrillée de taille 12*12.

L'agent apprend à se cacher à l'aide de reinforcement learning tandis que l'ennemi reste immobile.
Pour ce faire, nous utilisons l'implémentation de l'approche de Deep Q-Learning DQN basée sur le framework stable baselines 3 (SB3).

### Notions Principales

#### Map Quadrillée

Notre jeu se base sur une map quadrillé.  
Plus précisemment, nous considérons une map de jeu comme une matrice 12*12 composées de 7 valeurs différentes:
* 1 = case vide
* 2 = block infranchissable
* 3 = hors des limites de la map (utilisé dans certains cas)
* 4 = agent, joueur essayent de se cacher
* 5 = ennemi
* 6 = case vide mais dans la vision de l'ennemi
* 7 = point d'intérêt pour l'agent

Par exemple, la map suivante, en format "lisible" par un humain:  
![Image](/readme_imgs/CFV_ex1.png)  

est représentée par la matrice suivante:    
![Image](/readme_imgs/CFV_ex0.png)

#### Vision des Unités

Dans ce projet, nous considérons que les unités ont leur propre vision.  
Pour l'ennemi, cette vision est calculée à l'aide d'un algorithme A*.
Pour l'agent, puisque ce dernier se déplace à chaque étape, on utilise une méthode moins couteuse.

#### Point d'Intérêt

La notion de point d'intêret représente l'endroit qui semble être la plus intéressante pour l'agent à un instant t.
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
Exemple avec une map 4*4:  
![Image](/readme_imgs/CFV_ex1.png)

Si X représente l'agent, @ représente un block et $ le point d'intérêt: le vecteur d'observation sera (1, 1, SEP, 1, 2, SEP, 2) où SEP est une valeur de séparation (= -1)
        
#### FullDirectionOnFieldEnv (FDF)

Dans cette approche, les observations sont composés de cinq valeurs, représentant l' action la plus probable que l'agent doit effectuer pour serapprocher de son point d'intêret.
Plus précisément: des valeurs binaires représentant les actions possibles par l'agent (gauche, droite, haut, bas, arrêt).  
Exemple avec une map 4*4:  
![Image](/readme_imgs/CFV_ex1.png)
        
Si X représente l'agent, @ représente un block et $ le point d'intérêt: le vecteur d'observation sera (0, 1, 0, 0, 0)

#### SpiralFieldVisionEnv (SFV)

Dans cette approche, les observations sont composés d'un vecteur de 9*9 valeurs représentant les "alentours" de l'agent, ordonné en forme de spirale avec pour origine l'agent.
Un example d'ordre de spirale serait:  
![Image](/readme_imgs/SFV_ex1.png)
  
Avec l'exemple précédent:  
![Image](/readme_imgs/CFV_ex1.png)

Si X représente l'agent, @ représente un block et $ le point d'intérêt, et en considérant les représentations numériques des cases ( . = 1 , @ = 2, $ = 7):
le vecteur d'observation sera (1, 1, 1, 1, 2, 2, 1, 1, outside_value, outside_value, outside_value, outside_value, 1, 7, etc.)

#### FullFieldVisionEnv (FFV)

Dans cette approche, les observations sont une image RGB 32*32 représentant la map, le positionnement des joueurs, la vision de l'ennemi (perçue par l'agent) et le point d'intêret.
Pour ce faire, nous "triplons" le contenu de la matrice représentant la map et créons 3 channels représentant les channels RGB afin d'obtenir une image (les cases ont des valeurs différentes pour chaque channel).
  
Avec l'exemple précédent:  
![Image](/readme_imgs/CFV_ex1.png)
  
On obtiendrait une image "triplée":  
![Image](/readme_imgs/FFV_ex1.png)  

que l'on peut transformer en trois channels différents.

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

Les rewards sont simplistes :
* Si l'agent s'éloigne de son point d'intérêt, il est pénalisé
* Si l'agent se rapproche de son point d'intérêt, il est récompensé
* Si l'agent n'a plus de point d'intérêt, on considère qu'il est caché, il est largement récompensé

Nous avons expérimenté avec d'autres rewards, moins "intrusifs", afin d'étudier la possibilité pour l'agent d'apprendre à se cacher sans calcul explicite de son point d'intérêt. Plus d'information dans la partie Expérimentations

## Architecture
L'architecture du projet est présentée ci-dessous.

L'environnement abstrait d'un jeu hide & seek est décrit dans le dossier "envs" à l'aide des fichiers "hideSeekEnv.py", "maps.py" et "unit.py".
Cet environement abstrait est implémenté de plusieurs manières dans le dossier "envs/custom_envs"

Le dossier "data" contient les informations relatives aux données externes (notamment les maps, définissables au format textuel)

Le dossier "misc" contient des fonctions génériques

Le dossier "scripts" contient les scripts exécutables

Le dossier "learned_models" contient des modèles déjà appris (un pour chaque type d'environnement)

![Image](/readme_imgs/Archi.png)  

## Expérimentations

### Détails d'Apprentissage

Les modèles sont appris avec les paramètres par défaut de l'implémantation DQN de SB3, sauf le paramètre "exploration_fraction" qui est définit à 0.20 car cela permettait à l'agent une meilleure exploration des possibilités de mouvement dans notre cas.   

Les modèles sont tous appris sur la map "map_v4", avec plusieurs emplacements possibles pour l'ennemi et un placement statique (de départ) pour l'agent.  

Nous avons expérimenté avec des trainings de 200 000 à 2 000 000 steps.  
Selon nos résultats, les performances n'augmentent pas au delà de 700 000 ~ 900 000 steps.  

Les modèles de ./learned_models sont appris avec 700 000 steps.  

Dans le cas des approches CoordFieldVisionEnv, FullDirectionOnFieldEnv et SpiralFieldVisionEnv, "MlpPolicy" est utilisé en tant que policy.  
Dans le cas de l'approche FullFieldVisionEnv, "CnnPolicy" est utilisé en tant que policy.  

Les modèles appris avec ces paramètres sont indiqués par la mention "robust" dans le dossier /learned_model.

### Résultats

Voici les courbes de reward moyen par round de jeu (abscisse) sur le nombre d'étapes de jeu totale (ordonnée) pendant l'entraînement des modèles.  

CoordFieldVisionEnv  
![Image](/readme_imgs/CDF_correct_rewards.png)  

FullDirectionOnFieldEnv  
![Image](/readme_imgs/FDF_correct_rewards.png)  

FullFieldVisionEnv  
![Image](/readme_imgs/FFV_correct_rewards.png)  

SpiralFieldVisionEnv  
![Image](/readme_imgs/SVF_correct_rewards.png)  

Bien que ces courbes ne sont pas indicatives des performances d'un modèle, elles indiquent tout de même sa capacité à associer observations, rewards et actions.  
On peut notamment remarquer que le modèle basé sur l'environnement CoordFieldVisionEnv n'arrive pas à apprendre de manière effective comment maximiser les rewards.  

De manière générale, les environnements FullDirectionOnFieldEnv et FullFieldVisionEnv donnent (empiriquement) les meilleurs résultats.

### Exemples

Voici quelques exemples des deux modèles les plus performants: FullDirectionOnFieldEnv et FullFieldVisionEnv.  
Sur ces exemples: 'X' désigne l'agent qui se cache, 'Y' désigne l'ennemi. Un '0' désigne un block infranchissable, un '.' désigne une case vide et '-' désigne une case vide mais qui est vue par l'ennemi.  

Exemple de FullDirectionOnFieldEnv sur la map non vu pendant l'entrainement "map_v1"  
Start:  
![Image](/readme_imgs/FDF_ex2end.png)  
End:  
![Image](/readme_imgs/FDF_ex2trueend.png)  
Autre exemple,  
Start:  
![Image](/readme_imgs/FDF_ex1start.png)  
End:  
![Image](/readme_imgs/FDF_ex1end.png)  

Exemple de FullFieldVisionEnv sur la map non vu pendant l'entrainement "map_v1"  
Start:  
![Image](/readme_imgs/FFV_ex2start.png)  
End:  
![Image](/readme_imgs/FFV_ex2end.png)  
Autre exemple,  
Start:  
![Image](/readme_imgs/FFV_ex3start.png)  
End:  
![Image](/readme_imgs/FFV_ex3end.png)  

Nos modèles sont cependant loin d'être parfait, comme le montre cet exemple de l'environnement FullDirectionOnFieldEnv:
Start:  
![Image](/readme_imgs/FDF_badexstart.png)  
End:  
![Image](/readme_imgs/FDF_badexed.png)  
Ici, l'agent voit la case à droite de l'ennemi (Y) comme étant une case "libre", puisqu'il ne sait pas que la vision de l'ennemi s'y porte également.  
Il essaye alors de s'y rendre, mais ne peux pas traverser l'ennemi et s'arrête donc ici.

### Expérimentations de différent rewards

Ces rewards visent à étudier la capacité de l'agent à apprendre à se cacher de lui même.  

Les rewards testés sont "moins intrusifs" que les calculs à base de points d'intérêts:
* Si l'agent effectue un déplacement interdit (contre un block, l'ennemi ou essaye de sortir des limites de la map): il est pénalisé
* Si l'agent n'est pas dans le champ de vision de l'ennemi, il est récompensé
* Si l'agent se positionne contre plus d'un seul block, il est considéré comme "bien caché" et est récompensé
* Si l'agent se place de manière à ce qu'il y a plusieurs blocks entre lui et l'ennemi, il est récompensé

Globalement, les performances des modèles associés à ces rewards chutent comparés au rewards basé sur les points d'intérêt.  
Voici les courbes de reward moyen par round de jeu:

CoordFieldVisionEnv  
![Image](/readme_imgs/CDFmap4.png)  

FullDirectionOnFieldEnv  
![Image](/readme_imgs/FDFmap4.png)  

FullFieldVisionEnv  
![Image](/readme_imgs/FFVmap4.png)  

SpiralFieldVisionEnv  
![Image](/readme_imgs/SDFmap4.png)  

De manière similaire aux précédents rewards, on peut voir que le modèle basé sur l'environnement CoordFieldVisionEnv n'arrive pas à apprendre de manière effective comment maximiser les rewards.  

Voici quelques exemples de rounds utilisant ces rewards:
Exemple de CoordFieldVisionEnv sur la map "map_v4"  
Start:  
![Image](/readme_imgs/exCFVstartend.png)  
End:  
![Image](/readme_imgs/exCFVstartend.png)  

On voit que malgré le fait que le modèle soit appris sur cette map, il n'arrive pas à se cacher de manière effective.  

Exemple de FullDirectionOnFieldEnv sur la map non vu pendant l'entrainement "map_v1"  
Start:  
![Image](/readme_imgs/exFDFstart.png)  
End:  
![Image](/readme_imgs/exFDFend.png)  


Exemple de SpiralFieldVisionEnv sur la map non vu pendant l'entrainement "map_v1"  
Start:  
![Image](/readme_imgs/exSFVstart.png)  
End:  
![Image](/readme_imgs/exSFVend.png)  

Exemple de limitation de notre approche: environnement FullFieldVisionEnv sur la map d'entrainement "map_v4"  
Start:  
![Image](/readme_imgs/limitation1startFFV.png)  
End:  
![Image](/readme_imgs/limitation1endFFV.png)  

On voit que dans plusieurs cas, l'agent arrive tout de même à se cacher correctement.  
Cependant, de plus amples efforts seraient à fournir dans cette direction (tuning des hyperparamètres, etc.) afin d'obtenir des résultats comparables à l'approche à base de points d'intérêts.

Les modèles appris avec ces paramètres sont indiqués par la mention "experimental" dans le dossier /learned_model.

## Setup

Afin de faire fonctionner nos modèles, le package SB3 doit être installé:
```
!pip install "stable-baselines3[extra]>=2.0.0a4"
```

Pour lancer des train/evaluations, il suffit de cloner le repo:  
```
git clone https://github.com/lucassA/DeepRL.git
```
Rentrer dans le répertoire crée:  
```
cd DeepRL
```
Lancer le fichier main.py:  
```
python3 main.py -h
```
Exemple de train pour un environnement FullDirectionOnFieldEnv (les modèles de ./learned_models sont appris avec ces paramètres):
```
python3 main.py -env direction -a train -pmodel ./learned_models/new_models -pmap ./data/map_v4 -ep moves -lg True -o True
```
Exemple d'evaluation pour un environnement FullDirectionOnFieldEnv et un modèle appris ./learned_models/SFV_map4.zip (attention, le paramètre '-o' ne doit être activé que pour les 4 map originales):  
```
python3 main.py -env direction -a eval -pmodel ./learned_models/FDF_robust.zip -pmap ./data/map_v3 -ep moves -o True
```
