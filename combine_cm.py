import pandas as pd

cm_path = 'G:\\My Drive\\colab_data\\KG_GCN_GAN\\all_confusion_matrix_hmdb51.xlsx'
xls = pd.ExcelFile(cm_path)
df1 = pd.read_excel(xls, '8')
print(df1[:])
