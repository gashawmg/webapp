# import the necessary libraries

from distutils.command.upload import upload
import numpy as np
import pandas as pd
from matplotlib import image, pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
import streamlit as st
import pickle
from PIL import Image
import base64
import io

#--------- Use trained lgbm regressor  and HistGradientBoostingRegressor for predicting aqueous solubility of organic compounds
with open('model_lgbm.pkl','rb') as f:
         model1 = pickle.load(f)
with open('model_gbr.pkl','rb') as f:
          model2 = pickle.load(f)
#----------------------------------------------------------------------
st.set_page_config(page_title='Solubility Prediction App',layout='wide')
st.sidebar.markdown('<h2 style="color:#5a03fc;background-color:powderblue;border-radius:10px;text-align:center"> Use this Sidebar for Solubility Prediction </h2>',unsafe_allow_html=True)
st.markdown('<h3 style="color:#5a03fc;background-color:powderblue;border-radius:10px;text-align:center"> Introduction to Solubility Prediction App</h3>',unsafe_allow_html=True)
# Display my linkedin page on the sidebar and main page
st.sidebar.markdown("""[Gashaw M.Goshu](https://www.linkedin.com/in/gashaw-m-goshu/), Ph.D in Organic Chemistry""")
st.markdown("""[Gashaw M.Goshu](https://www.linkedin.com/in/gashaw-m-goshu/), Ph.D in Organic Chemistry""")

# Define solubility and explain why it is important
st.markdown("""`Solubility` is defined as the maximum amount of solute that will dissolve in a given amount of solvent to form a saturated solution at a specified temperature, usually at room temperature. Aqueous solubility is one of the important properties in drug discovery. As a result, solubility of compounds needs to be estimated before synthesizing them. This Web App was developed by training 6,960 data points (70% of 9,943) using 40 algorithms. Best results were obtained by two models (Light GBM Regressor(LGBMR) and Histogram-based Gradient Boosting Regressor(HGBR)). See the 10-fold cross-validation results shown below.""")

figure1 = Image.open('figure1.JPG')
st.image(figure1, caption='Figure 1. 10-fold cross-validation using Light GBM Regressor on training set')
figure2 = Image.open('Figure2.JPG')

st.image(figure2, caption='Figure 2. 10-fold cross-validation using HistGradient Boosting Regressor on training set')
 
st.markdown("""After tunning parameters, the prediction of LGBMR and HGBR on a test data was averaged to give the performance on the whole test dataset( 2,983)  shown in the following **figure**.""")

# Plot the test data using the predict and actual values
test = pd.read_csv('test_2983.csv')

# model performance using RMSE
rmse = np.sqrt(mean_squared_error(test['Actual'], test['Predicted']))  

# R^2 (coefficient of determination) regression score function: 
R2 =r2_score(test['Actual'], test['Predicted'])

# Plot the figure of the test dataset on the webpage
sn.regplot(x=test['Predicted'] , y=test['Actual'],line_kws={"lw":2,'ls':'--','color':'black',"alpha":0.7})
plt.xlabel('Predicted logS(mol/L)', color='blue')
plt.ylabel('Experimental logS(mol/L)', color ='blue')
plt.title("Test dataset", color='red')
plt.grid(alpha=0.2)
R2 = mpatches.Patch(label="R2={:04.2f}".format(R2))
rmse = mpatches.Patch(label="RMSE={:04.2f}".format(rmse))
plt.legend(handles=[R2, rmse])
st.pyplot(plt)

# ============ User input
data = st.sidebar.text_input('Enter SMILES in single or double quotation separated by comma:',"['CCCCO']")
st.sidebar.markdown('''`or upload SMILES strings in CSV format, note that SMILES strings of the molecules should be in 'SMILES' column:`''')
multi_data = st.sidebar.file_uploader("=====================================")
X_train = pd.read_csv("X_train.csv")

st.sidebar.markdown("""**If you upload your CSV file, click the button below to get the solubility prediction** """)
prediction = st.sidebar.button('Predict logS of molecules')

# ================= Get the names of the 200 descriptors from RDKit
def calc_rdkit2d_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    # Append 200 molecular descriptors to each molecule in a list
    Mol_descriptors =[]
    for mol in mols:
        # Calculate all 200 descriptors for each molecule
        mol=Chem.AddHs(mol)
        descriptors = np.array(calc.CalcDescriptors(mol))
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names  

# ================= Use only 71 molecular descriptors
features = ['MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'MolWt',
       'FpDensityMorgan2', 'BalabanJ', 'HallKierAlpha', 'Ipc', 'Kappa3',
       'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
       'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4',
       'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA9', 'SMR_VSA10',
       'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA6', 'SMR_VSA7',
       'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA3',
       'SlogP_VSA4', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA10',
       'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
       'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
       'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
       'VSA_EState6', 'VSA_EState8', 'VSA_EState9',
       'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
       'NumAliphaticRings', 'NumAromaticHeterocycles', 'MolLogP',
       'fr_Al_COO', 'fr_Al_OH', 'fr_NH0', 'fr_NH1', 'fr_NH2',
       'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_aniline',
       'fr_bicyclic', 'fr_ether', 'fr_halogen', 'fr_methoxy',
       'fr_para_hydroxylation']
#============ A function that can generate a csv file for output file to download
# Reference: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/2
#           https://github.com/dataprofessor/ml-auto-app/blob/main/app.py
def filedownload(data,file):
    df = data.to_csv(index=False)
    f= base64.b64encode(df.encode()).decode()
    link = f'<a href ="data:file/csv; base64,{f}" download={file}> Download {file} file</a>'
    return link

if data!= "['CCCCO']":
    df = pd.DataFrame(eval(data), columns =['SMILES'])
    #========= function call to calculate 200 molecular descriptors using SMILES
    Mol_descriptors,desc_names = calc_rdkit2d_descriptors(df['SMILES'])
    #========= Put the 200 molecular descriptors in  table
    Dataset_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
    #========= Use only the 71 descriptors listed above
    X_test = Dataset_with_200_descriptors[features]
    #======== The data was standardized during traning and test set also need to be standardized
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #---------------------------------------------------------------------

    #======== Prediction of solubility using model1(LightGBM) and model2(HistGradientBoostingRegressor)
    X_logS = (model1.predict(X_test) + model2.predict(X_test))/2

    #======= Put the predicted solubility in Dataframe
    predicted = pd.DataFrame(X_logS, columns =['Predicted logS (mol/L)']) 

    #======= Concatenate SMILES and the predicted solubility 
    output = pd.concat([df,predicted], axis=1)
    st.sidebar.markdown('''## See your output in the following table:''')
    #======= Display output in table form
    st.sidebar.write(output)

    #======= show CSV file attachment
    st.sidebar.markdown(filedownload(output,"predicted_logS.csv"),unsafe_allow_html=True)

#===== Use uploaded SMILES to calculate their logS values
elif prediction:
     df2 = pd.read_csv(multi_data)
     Mol_descriptors,desc_names = calc_rdkit2d_descriptors(df2['SMILES'])
     Dataset_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
     X_test = Dataset_with_200_descriptors[features]
     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)
     X_logS = (model1.predict(X_test) + model2.predict(X_test))/2
     predicted = pd.DataFrame(X_logS, columns =['logS (mol/L)'])
     output = pd.concat([df2['SMILES'],predicted], axis=1)
     st.sidebar.markdown('''## Your output is showm in the following table:''')
     st.sidebar.write(output)
     st.sidebar.markdown(filedownload(output,"predicted_logS.csv"),unsafe_allow_html=True)
  
else:
    st.markdown('<p style="color:#5a03fc;background-color:lightblue;border-radius:10px;text-align:center"> This App accepts SMILES strings. If you have few molecules, you can directly put the SMILES in a single or double quotation separated by comma in the sidebar as shown in the following screenshot below.</p>',unsafe_allow_html=True)
    inputbox1 = Image.open('inputbox.JPG')
    st.image(inputbox1, caption='Put SMILES in input box',width=300)

    st.markdown('<p style="color:#5a03fc;background-color:powderblue;border-radius:10px;text-align:center"> If you press enter, you will get the following output or you can download your output using the download link. Note that this App does not store or save your data to the server and the download link will be removed if you refresh the app.</p>',unsafe_allow_html=True)

    inputbox2 = Image.open('input_box_output.JPG')
    st.image(inputbox2, caption='Output file should look like this',width=300)

    st.markdown('<p style="color:#5a03fc;background-color:powderblue;border-radius:10px;text-align:center"> If you have many molecules, you can put their SMILES strings in a "SMILES" column, upload them and click the button which says "Predict logS of moleclues" </p>',unsafe_allow_html=True)

    smiles = Image.open('SMILES_column.JPG')
    st.image(smiles, caption='1) Put SMILES in CSV file',width=300)

    upload_smiles = Image.open('browse_files.JPG')
    st.image(upload_smiles, caption='2) Click Browse files button and your SMILES csv file',width=200)

    predict_button = Image.open('predict_logS.JPG')
    st.image(predict_button, caption='3) Click this button to get your prediction',width=200)

    outpufile = Image.open('outputfile.JPG')
    st.image(outpufile, caption='4) Your output file should look like this',width=200)      
