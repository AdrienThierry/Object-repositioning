##################################################
I. DEPENDANCES
##################################################

- OpenCV
- SDL2

##################################################
II. COMPILATION (si l'exécutable ne marche pas)
##################################################

cmake .
make

##################################################
III. UTILISATION
##################################################

Syntaxe : ./object_repositioning chemin_image_a_charger

0) Attendre quelques secondes que l'image s'affiche (le meanshift se fait au lancement du programme)

1) Cliquer à l'endroit où vous souhaitez placer la ligne qui marque la profondeur
maximale de la scène

2) Dessiner une boîte englobante autour de l'objet à segmenter

3) Attendre (~1 min). Quand les calculs sont finis, seul l'objet extrait est affiché.

4) Les touches du pavé numérique permettent d'accéder au différents modes de visualisation :
	0 -	Mode interactif. Vous pouvez faire un drag&drop sur l'objet choisi pour en déplacer
		une copie dans la scène.
	1 -	Résultat du MeanShift
	2 -	Classification des superpixels selon leur intersection avec la boîte englobante.
		* En blanc : superpixels à l'extérieur de la boîte englobante
		* En noir : superpixels qui intersectent la boîte englobante
		* En gris : superpixels à l'intérieur de la boîte englobante
	3 -	Carte de salience
	4 - Probabilité d'appartenance à l'arrière plan
		(intensité du pixel proportionnelle à la probabilité)
	5 - Probabilité d'appartenance à l'avant plan
		(intensité du pixel proportionnelle à la probabilité)
	6 -	Terme de "smoothness" pour le graph cut (poids entre un pixel et son
		voisin du bas)
	7 -	Terme de "smoothness" pour le graph cut (poids entre un pixel et son
		voisin de droite)
	8 - Résultat de la segmentation


##################################################
IV. LICENCES
##################################################

Code du Graphcut
----------------
Copyright 2001-2006 Vladimir Kolmogorov (v.kolmogorov@cs.ucl.ac.uk), Yuri Boykov (yuri@csd.uwo.ca).

This software can be used for research purposes only.
    If you require another license, you may consider using version 2.21
    (which implements exactly the same algorithm, but does not have the option of reusing search trees).

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Fichiers concernés :
- instances.inc
- maxflow.cpp
- block.h
- graph.cpp
- graph.h

Code du Meanshift
-----------------
Kanglai Qian (https://github.com/qiankanglai/opencv.meanshift)

Fichiers concernés
- MeanShift.cpp
- MeanShift.h
