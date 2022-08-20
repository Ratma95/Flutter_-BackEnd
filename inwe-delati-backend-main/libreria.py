import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import preprocessing

#Retorna la data de la base de datos como un DataFrame
#Ingresa la query y la conexion
def get_dataFrame(sql, con):
    df = pd.read_sql_query(sql, con = con)
    return df

#Transforma la data del DataFrame a numeros
#Ingresa el DataFrame a transformar
def transform_dataFrame(dat):
    label_encoder = preprocessing.LabelEncoder()
    transformed_data = dat.apply(label_encoder.fit_transform)
    return transformed_data

#Transforma la data del DataFrame a numeros de 0 a 1
#Ingresa el DataFrame a transformar
def normalization_dataFrame(dat):
    dfo_transformed=(dat-dat.min())/(dat.max()-dat.min())
    return dfo_transformed

#Ingresa el DataFrame de la data normalizada,el clustering model con los clusteres,el numero de componentes(2 o 3)
#Retorna Dataframe de componentes normalizada con una columna Cluster
def acp(dfo_transformed,clustering_model,num_componentes):
    #Clusteres de cada fila
    predicted_labels=clustering_model.labels_  
    #Creamos el PCA segun el numero de componentes
    pca=PCA(n_components=num_componentes)
    #Ajustamos la data
    pca_demanda=pca.fit_transform(dfo_transformed)
    #Creamos la tabla de componentes
    if num_componentes==3:
        pca_demanda_df = pd.DataFrame(data=pca_demanda, columns=['Componente_1','Componente_2','Componente_3'])
    else:
        pca_demanda_df = pd.DataFrame(data=pca_demanda, columns=['Componente_1','Componente_2'])
    #Normalizamos la tabla de componentes
    pca_demanda_df=normalization_dataFrame(pca_demanda_df)
    #Añadimos la columna de cluster a la tabla de componentes 
    pca_df=pd.concat([pca_demanda_df,pd.DataFrame(data=predicted_labels,columns=['Cluster'])], axis=1)

    return pca_df

#Ingresa la DataFrame creado por acp para dos componentes
#Retorna la imagen 2D
def imagen2D (pca_df,titulo): 
    #Columna de Cluster a lista
    clusters=pca_df['Cluster'].values
    #Clusteres labels
    cluster_labels=set(clusters)
    #Numero de clusters
    n_clusters= len(cluster_labels)
    #Inicializamos la imagen
    fig=plt.figure(figsize=(12,12))
    ax=fig.add_subplot(1,1,1)   
    #Dibujamos las caracteristicas basicas de la imagen   
    ax.set_xlabel('Componente_1', fontsize=15)
    ax.set_ylabel('Componente_2', fontsize=15)
    ax.set_title(titulo, fontsize=20)
    #Dibujamos los puntos
    for cluster in cluster_labels:
        #Color
        color = cm.nipy_spectral(float(cluster+1) / n_clusters)  
        if (cluster) != -1:      
            ax.scatter(pca_df.iloc[clusters== (cluster), 0], pca_df.iloc[clusters==(cluster), 1], s=80, c=np.array([color]),label=f"Cluster {cluster}")
        else:      
            ax.scatter(pca_df.iloc[clusters== (cluster), 0], pca_df.iloc[clusters==(cluster), 1], s=5, c='Gray',label=f"Outlier")
    #Legenda
    fig.legend(title='Clusters', loc='upper left', fontsize='small')
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.6)
    #Creamos el jpg
    my_stringIObytes = io.BytesIO()
    fig.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData=base64.b64encode(my_stringIObytes.read())
    plt.close(fig)
    return my_base64_jpgData.decode()

#Ingresa la DataFrame creado por acp para tres componentes
#Retorna la imagen 3D
def imagen3D (pca3d_df,titulo):
    #Columna de Cluster a lista
    clusters=pca3d_df['Cluster'].values
    #Clusteres labels
    cluster_labels=set(clusters)
    #Numero de clusters
    n_clusters= len(cluster_labels)
    #Inicializamos la imagen
    fig3d=plt.figure(figsize=(13,12))
    ax3d=fig3d.add_subplot(1,1,1,projection='3d')
    #Dibujamos las caracteristicas basicas de la imagen  
    ax3d.set_xlabel('Componente_1', fontsize=15)
    ax3d.set_ylabel('Componente_2', fontsize=15)
    ax3d.set_zlabel('Componente_3', fontsize=15)
    ax3d.set_title(titulo, fontsize=20)
    #Dibujamos los puntos
    for cluster in cluster_labels:
         color = cm.nipy_spectral(float(cluster+1) / n_clusters) 
         if (cluster) != -1:      
            ax3d.scatter(pca3d_df.iloc[clusters== (cluster), 0], pca3d_df.iloc[clusters==(cluster), 1], pca3d_df.iloc[clusters==(cluster),2], s=80, c=np.array([color]),label=f"Cluster {cluster}")
         else:
            ax3d.scatter(pca3d_df.iloc[clusters== (cluster), 0], pca3d_df.iloc[clusters==(cluster), 1], pca3d_df.iloc[clusters==(cluster),2], s=5, c='Gray',label=f"Outlier")
    #Legenda
    fig3d.legend(title='Clusters', loc='upper left', fontsize='small')
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.6)
    #Creamos el jpg
    my_stringIObytes = io.BytesIO()
    fig3d.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData=base64.b64encode(my_stringIObytes.read())
    plt.close(fig3d)
    return my_base64_jpgData.decode()

#Ingresa el dataframe de los valores transformados como una lista de listas y el dbscan
#Devuelve los puntos core de cada cluster
def puntosCoreCluster(dataframe_list,dbscan):
    #Lista de listas de los valores de las filas del dataframe
    puntos=dataframe_list
    #Cluster de cada lista
    puntos_clusters=list(dbscan.labels_)
    #Valores de las filas con su cluster
    lista_puntos_clusters=[]
    #Adicionamos elementos a lista_puntos_clusters
    for x in range(len(puntos)):
        lista_puntos_clusters.append([puntos[x],puntos_clusters[x]])
    #Lista de core con formato lista_puntos_clusters
    lista_core=[]
    #Indices de core del dataframe
    puntos_core=list(dbscan.core_sample_indices_)
    #Llenamos la lista_core
    for cores in puntos_core:
        #Obtenemos el elemento de lista_puntos_clusters con el indice core
        temporal=lista_puntos_clusters[cores]
        #Adicionamos el indice a el elemento de lista_puntos_clusters
        temporal.append(cores)
        #Agregamos el elemento a la lista_core
        lista_core.append(temporal)
    #Creamos una key por cluster
    dict_core={}
    clusters=set(list(dbscan.labels_))
    for cluster in clusters:
        if(cluster!=-1):
            dict_core[str(cluster)]=[]
    #Agregamos los elementos de la lista_core a su respectivo cluster
    for i in range(len(lista_core)):
        temporal=dict_core[str(lista_core[i][1])]
        temporal.append([lista_core[i][0],lista_core[i][2]])
        dict_core[str(lista_core[i][1])]=temporal
    return dict_core
