Für das Ausführen des Evaluations -und Trainingsskripts wird Jupyter Notebook benötigt. Sicherheitshalber sollte der Kernel nach jedem Durchlauf neu gestartet werden, da es in einigen Fällen zu Fehlermeldungen kommen kann.
Jeder Block in Notebook sollte sequenziell ausgeführt werden.


# Wichtige Variablen zum Ausführen

- Die Einstellungen, die in der settings.json gespeichert werden(batch_size, epochs, learn_rate usw.)
- path_save_model gibt das Verzeichnis an, in dem das trainierte Model gespeichert wird.
- Pfade zum Trainingsdatensatz und zum Validierungsdatensatz mit den entsprechenden Labels
- Für das Trainingsskript gelten dieselben Anforderungen

Achtung in der requirements.txt: Beim Python-Modul pytorch-cuda sollte an die Hardware und die Version der installierten CUDA-API angepasst werden.


# Übergabeparameter für InferenceEnet.py

-ds/--dataset           Mit diesem Flag wird das Model übergeben, welches mit einem bestimmten Datensatz trainiert wurde. (riverblindness oder schistosoma)

-i/--imagePath          Gibt den Pfad zum Eingabebild an.
-vo                     Gibt die angewendete Visualisierungsoption an. (contour, alpha, paintBackground)
-a                      Gibt den Transparenzwert für die Transparenzvisualisierung an.
-r                      Gibt an in welchem Format das Endresultat gespeichert werden soll. Bei True ist das normal Resultat. Bei False wird ein Plot mit den Verarbeitungsschritten geplottet.

-at/--areaThreshold     Gibt den Wert für den Flächengrenzwert der primären Objekte an.

# Skripte für die Analyse der Metriken

path_metrics gibt den Pfad zum log Verzeichnis des Evaluations -und Trainingsdurchlaufsa an.


