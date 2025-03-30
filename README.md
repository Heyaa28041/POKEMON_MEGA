# POKEMON_MEGA
Question: Your task is to determine whether a Pokémon is a Mega Evolution or not using the dataset available at the following Kaggle link: Dataset.
Objective
The goal is to use machine learning to classify each Pokémon as either a Mega Evolution or a Regular Pokémon, without using the Pokémon name as a feature. The model should rely only on statistical attributes (e.g., HP, Attack, Defense, etc.) for the classification. Additionally, you should evaluate the model’s performance by plotting metrics such as:
Confusion matrix


ROC curve


Precision-recall curve
The final output should be saved in CSV format with two columns:
Pokemon: The name of the Pokémon (for identification purposes only in the output).


Mega_Evolution: Yes if the Pokémon is a Mega Evolution, No otherwise.


Example Output (CSV format)
Pokemon,Mega_Evolution  
Charizard,No  
Mega Charizard X,Yes  
Blastoise,No  
Mega Blastoise,Yes  
Gengar,No  
Mega Gengar,Yes  
Pikachu,No  
Lucario,No  
Mega Lucario,Yes 
Answer: 
STEP 1:
Download and exract the zip file from github.
STEP 2: 
Download the dataset from kaggle and place it in the directory of the repositry and name it as Pokemon1.csv.
Step 3:
Install the libraries mentioned in the file or use the command pip install pandas scikit-learn matplotlib seaborn.
Step 4:
Run the Python Script after ensuring that the csv pokemon1 file and the python files are in the same path. 
Step 5: 
After running,, the script will save another csv file for predictions and open Confusion matrix, ROC curve, Precision-recall curve.
Thank You


