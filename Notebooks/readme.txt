Please note that all Jupyter Notebooks within this folder were run in google colab. 
They will run within colab but might not if run locally. To run them yourself, 
first upload the files to google colab. 

Then use 1_Data_Preprocessing to create the JRC-Aquis dataset. 
This notebook provides a script which downloads and preprocesses 
the JRC-Acquis files. These can be saved either locally or using google drive using 
----
from google.colab import drive
drive.mount("/content/gdrive")
df_merged.to_pickle(<some Directory>)
----
and then specifying a diretory within google drive.
-
Notebooks 2,3 and 4 load files from my personal google drive. 
I would provide the files here but they are incovienently large.  
Therefore, use the first Notebook to create the dataset. Then save it by
following the instructions above. 
The directories used for 2,3, 4 is 
"/content/gdrive/My Drive/NCC/EU-BookShop Files/"
or respectively 
"/content/gdrive/My Drive/NCC/JRC_Arquis_files/"
if another directory is used, change the variable "main_dir" accordingly in the other notebooks.
