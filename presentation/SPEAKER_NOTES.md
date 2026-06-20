# Notes orateur : ce qu'il faut dire à chaque slide

Présentation Minecraft Mob Vision, 19 slides, vise ~10 min. La classe connaît Minecraft, le prof non : une phrase de contexte de temps en temps suffit. Idée directrice : à chaque slide, dire *quel problème on résout* avant de montrer *comment*. Les chiffres clés à connaître par coeur : **96,3% top-1 · 99,1% top-5 · 84,1% IoU · 87 classes · 22 809 images**.

Conseil de rythme : ne pas lire les slides. Les puces sont des repères, tu développes à l'oral. Vise ~30 s par slide de contenu, un peu plus sur les slides 3, 5, 10, 16 (les moments importants).

---

## Slide 1 : Titre

- Te présenter, annoncer le sujet en une phrase : "reconnaître automatiquement quelle créature de Minecraft est à l'écran, et l'encadrer d'une boîte."
- Donner les 3 chiffres tout de suite, ça pose le niveau : 96% de bonne créature, 99% dans le top 5, et la boîte recouvre la vraie à 84%.
- Enchaîner : "je vais vous montrer comment, en partant des données jusqu'au modèle final."

## Slide 2 : Le problème

- Pour le prof : "Minecraft est un jeu fait de cubes ; un *mob* c'est une créature, animal ou monstre ; une *frame* c'est juste une image du jeu."
- Deux tâches en même temps : **quoi** (classer parmi 87 créatures) et **où** (prédire un rectangle, des nombres continus, donc de la régression).
- Le point qui rapporte des points : on fait les deux d'un coup avec UN seul réseau, c'est le volet "détection d'objets", le niveau au-dessus de la simple classification.
- Préciser le format de la boîte : centre (cx, cy) + taille (w, h), tout entre 0 et 1. C'est le standard YOLO, on le reverra.

## Slide 3 : Un dataset synthétique auto-labélisé

- C'est LA particularité du projet, prends ton temps. Problème classique : pour entraîner un détecteur il faut dessiner des milliers de rectangles à la main, c'est long et plein d'erreurs.
- Ma solution : un mod (extension du jeu) que j'ai codé et mis en open-source, qui génère ET étiquette tout automatiquement.
- Comment les boîtes sont exactes : le jeu connaît déjà la boîte de collision 3D de chaque créature, je la projette à l'écran. Donc zéro erreur d'étiquetage, contrairement à l'annotation humaine.
- Le graphe : 22 800 images, 87 créatures, et c'est bien équilibré (~260 par classe, à peine 1,6x d'écart). Donc pas besoin de tricks de rééquilibrage.

## Slide 4 : Vérité-terrain exacte

- Montrer les exemples : "tous ces rectangles oranges sont posés automatiquement par le programme, aucun humain n'a cliqué."
- Insister : comme les boîtes sont parfaites, la seule difficulté qui restera pour le modèle, c'est quand deux créatures se ressemblent vraiment. On le reverra dans l'analyse d'erreurs.

## Slide 5 : La caméra orbite

- Expliquer concrètement comment le mod marche : la créature reste fixe, la caméra tourne autour en photographiant à chaque petit pas d'angle.
- Montrer la bande : "c'est le même dragon, vu sous tous les angles, et la boîte le suit parfaitement à chaque image."
- Conclure : c'est ça qui rend le dataset à la fois énorme et parfaitement étiqueté pour zéro effort manuel. Le mod tourne seul pendant des heures.

## Slide 6 : EqualizeV (préprocessing)

- Le problème : dans Minecraft il y a un cycle jour/nuit, donc la même créature est claire à midi et quasi noire la nuit. Cette luminosité ne dit rien sur l'identité, c'est du bruit qui peut tromper le modèle.
- La solution : on égalise la luminosité de chaque image (techniquement le canal V en HSV) tout en gardant les couleurs intactes.
- Montrer le avant/après sur les images de nuit : on récupère les détails sans changer les couleurs. Et c'est fait une seule fois au pré-calcul, pas à chaque epoch.

## Slide 7 : Augmentations

- Idée : créer artificiellement de la variété pour que le modèle généralise mieux. On retourne l'image, on bricole un peu couleurs et contraste.
- Le mot-clé : augmentations **légères**. On ne change jamais ce qu'est la créature (sinon on casse l'étiquette).
- Bonus technique (si le temps) : un cache sur disque où on décode les images une seule fois, ça rend l'entraînement bien plus rapide, limité par le GPU et plus par le disque.

## Slide 8 : Architecture (MobDetector)

- Un seul réseau : un tronc commun (le "backbone", un réseau déjà pré-entraîné sur des millions d'images) puis deux têtes, une pour la classe, une pour la boîte.
- Détail malin : je garde la carte spatiale en sortie du tronc (au lieu de la moyenner), ce qui aide à localiser ET à visualiser l'attention plus tard.
- Le tronc est interchangeable depuis un simple fichier de config, sans toucher au code : c'est ce qui m'a permis de comparer plusieurs réseaux facilement. Modèle compact, ~16 millions de paramètres.

## Slide 9 : Stratégie d'entraînement

- Transfer learning : le tronc sait déjà voir des formes, on ne réapprend pas de zéro.
- En deux temps : d'abord on gèle le tronc et on entraîne juste les têtes neuves, puis on dégèle tout doucement (avec un taux d'apprentissage 10x plus faible sur le tronc) pour ne pas casser ce qu'il savait.
- La loss de la boîte est une CIoU : elle optimise le rectangle dans son ensemble (recouvrement, distance des centres, forme), bien mieux qu'une simple distance coordonnée par coordonnée. Reste technique, ne pas s'éterniser.

## Slide 10 : Comparaison des backbones

- Montrer que j'ai vraiment exploré : EfficientNet, puis MobileNetV4 small et medium, puis ConvNeXtV2.
- Être honnête (ça fait sérieux) : sur le test pur, le MobileNetV4-medium est même un poil meilleur et plus petit. Je ne le cache pas.
- Mais je garde le ConvNeXtV2 pour deux choses que le score ne montre pas : une carte d'attention beaucoup plus nette (slide d'après) et une meilleure tenue sur de vraies images d'internet. Message : un bon choix de modèle, ce n'est pas juste une ligne de score.

## Slide 11 : Résultats

- Les chiffres clés en grand. 96,3% de bonne classe sur des images jamais vues, parmi 87 possibilités, et 84% de recouvrement moyen des boîtes.
- Resituer : c'est solide pour un modèle aussi compact, et c'est mesuré sur un vrai split de test mis de côté.

## Slide 12 : Robustesse

- Le truc rassurant : ça ne s'effondre dans aucune condition. Météo, heure du jour, taille de la créature, distance : on reste partout entre 95 et 98%.
- Ce que ça prouve : le modèle ne triche pas en exploitant un détail du décor ou de la lumière, il marche vraiment partout. Pas d'angle mort.

## Slide 13 : Analyse d'erreurs

- L'intéressant, c'est COMMENT il se trompe. Et toutes les erreurs sont des sosies.
- Donner les exemples du tableau : ghast vs happy_ghast (même corps blanc, juste la tête change), vache vs mouton, têtard vs poisson tropical.
- C'est exactement le genre d'erreur qu'on veut : ça prouve qu'il a appris de vraies caractéristiques visuelles, il ne se plante que sur ce qui est dur même pour un humain au premier coup d'oeil. Montrer les images : on comprend l'erreur en les voyant.

## Slide 14 : La longue traîne

- Les rares cas vraiment durs : des petits poissons quasi invisibles dans l'eau, ou des créatures rares.
- Même là, il reste majoritairement correct. Pas de classe catastrophique, juste une longue traîne attendue.

## Slide 15 : Interprétabilité

- On veut vérifier que le modèle regarde la bonne chose, pas le décor. On colore les pixels qui pèsent le plus dans sa décision.
- Détail technique si on me le demande : la méthode classique (Grad-CAM) ne marche pas bien sur ce réseau, j'utilise une variante (gradient x input) qui reste nette. Ne pas développer sauf question.
- Le résultat : l'attention est bien sur la créature. C'est la preuve visuelle derrière les 96%, pas juste un chiffre.

## Slide 16 : L'évolution de la heatmap

- Le slide le plus parlant, à mettre en valeur. Mêmes images, passées dans une vieille version du modèle (à gauche) et la version actuelle (à droite).
- Avant : une grosse tache diffuse étalée sur la moitié de l'image, le modèle regardait un peu partout. Maintenant : une zone petite et précise, pile sur la créature.
- Conclure : ça montre concrètement les progrès du projet entre les itérations, pas juste un chiffre qui monte. C'est aussi ce qui justifie le choix du ConvNeXtV2 du slide 10.

## Slide 17 : Prédictions sur de vraies images

- Le vrai test final : des screenshots Minecraft pris sur internet, donc différents de mes images générées (pas le même rendu, pas les mêmes décors).
- Le modèle sort la bonne classe, la confiance et le rectangle. Il tient bien hors de son terrain d'entraînement, c'est le signe qu'il a appris quelque chose de général, pas par coeur.

## Slide 18 : Conclusion

- Récap rapide : un dataset synthétique aux étiquettes parfaites, du transfer learning sur un petit réseau moderne, et des erreurs interprétables.
- Pistes honnêtes : gérer plusieurs créatures par image (là c'est une seule), améliorer les petits objets, et tester sur de vraies captures pour mesurer l'écart simulation / réel.

## Slide 19 : Merci

- Remercier, inviter aux questions.
- À garder sous le coude pour les questions : le notebook tourne en live si on veut une démo, la comparaison de modèles, l'histoire de la heatmap, et les deux repos open-source (code + générateur de dataset).
