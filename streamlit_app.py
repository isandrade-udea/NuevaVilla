import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
# Modelado y Forecasting
from statsmodels.tsa.stattools import acf
# from skforecast.ForecasterAutoreg import ForecasterAutoreg
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Borrar la caché en cada ejecución
st.cache_data.clear()

#python -m pip install {package_name}


# Definir una paleta de colores personalizada basada en la imagen
cootracovi_palette = [
                      "#FFD700",  # Amarillo o beige para cercanía y calidez
                      "#FFA07A"]  # Naranja suave para dinamismo y energía

custom_yellow_vibrant = sns.color_palette(["#FFEA00", "#FFB300", "#FF8C00", "#FF6F00"])

# Combinar las dos paletas en un solo degradado
combined_palette = cootracovi_palette + custom_yellow_vibrant

# Usar esta paleta combinada en un gráfico
sns.set_palette(combined_palette)

st.title(":tram: Nueva Villa ")
st.write(
    "modelos predictivo de los parametros de las rutas: 080, 081, 082 A y 082 B, del barrio Villa Hermosa"
)

# Cargar datos desde un enlace de GitHub
st.subheader("Cargar Datos desde un Link de GitHub")

# URL predeterminada
url_predeterminado = "https://raw.githubusercontent.com/isandrade-udea/LabIA/refs/heads/main/Transporte/NuevaVilla/INFORMACION%20VIAJE%20A%20VIAJE.xlsx%20-%20Hoja1.csv"

# Crear el campo de texto con la URL predeterminada
url = st.text_input("Introduce la URL del archivo CSV en GitHub", value=url_predeterminado)

def cargar_csv_desde_url(url):
    # Verificar si se ha introducido una URL
    if url:
        try:
            # Leer el archivo CSV directamente desde la URL
            df = pd.read_csv(url)
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return None
    else:
        st.write("Por favor, introduce la URL de un archivo CSV en GitHub para continuar.")
        return None

df = cargar_csv_desde_url(url)

# Mostrar las primeras filas del DataFrame
st.write("Datos Cargados:")
st.dataframe(df) # Muestra las primeras filas del DataFrame

#preprocesamiento

df.columns = df.columns.str.lower().str.replace(' ', '_')

df['fecha_hora_inicio'] = pd.to_datetime(df['fecha'] + ' ' + df['horainicio'])
df['fecha_hora_inicio'] = pd.to_datetime(df['fecha_hora_inicio'], format='%Y-%m-%d %H:%M:%S')


if 'unnamed:_9' in df.columns:
  df = df.drop('unnamed:_9', axis=1)
# borrar columna horafinal
if 'horafinal' in df.columns:
  df = df.drop('horafinal', axis=1)

if 'fecha' in df.columns:
  df = df.drop('fecha', axis=1)

if 'horainicio' in df.columns:
  df = df.drop('horainicio', axis=1)

# prompt: borrar filas con nulos
df = df.dropna()

df['duracion'] = df['duracion'].astype(float)
df['tiempoterminal'] = df['tiempoterminal'].astype(float)
df['kilometros_recorridos'] = df['kilometros_recorridos'].str.replace(',','.').astype(float)
df['ruta_marcada'] = df['ruta_marcada'].astype(str)

# Generar una columna para el día
df['dia'] = df['fecha_hora_inicio'].dt.day

# Crear la columna 'Hora' con la hora extraída de 'Fecha_Hora_Salida'
df.loc[:, 'hora'] = df['fecha_hora_inicio'].dt.hour

# Función para clasificar las horas en jornada, incluyendo la madrugada
def clasificar_jornada(hora):
    if 0 <= hora < 6:
        return 'Madrugada'
    elif 6 <= hora < 12:
        return 'Mañana'
    elif 12 <= hora < 18:
        return 'Tarde'
    else:
        return 'Noche'

# Aplicar la función para crear la columna 'Jornada'
df.loc[:, 'jornada'] = df['hora'].apply(clasificar_jornada)

# fijamos la columna como indice
df = df.set_index('fecha_hora_inicio')

df['dia_n'] = df.index.day_name() 

#organizamos en orden cronologico
df.sort_index(inplace=True)

df.columns = [col.replace('tiempoterminal', 'tiempo_terminal') for col in df.columns]


#Tamaño del dataset
st.write(f"El tamaño del dataset despues de el preprocesamiento es de: {df.shape[0]} filas y {df.shape[1]} columnas:")
st.write(", ".join(df.columns))

st.subheader('Analisis de las variables')
opciones_columnas = [
        'pasajeros', 
        'duracion', 
        'tiempo_terminal', 
        'kilometros_recorridos', 
        'num/vehiculo']

    # Selección de columna con 'tipo_negocio' como predeterminado
columna_seleccionada = st.selectbox(
        "Selecciona la columna para graficar:", 
        opciones_columnas, 
        index=opciones_columnas.index('num/vehiculo'),
        key='columna_seleccionada')


# Cálculos
valor_medio = round(df[columna_seleccionada].mean(), 2)
sesgo = round(df[columna_seleccionada].skew(), 2)
percentil_25 = df[columna_seleccionada].quantile(0.25)
percentil_75 = df[columna_seleccionada].quantile(0.75)
iqr = percentil_75 - percentil_25

# Cálculo de valores atípicos
outliers = df[(df[columna_seleccionada] < (percentil_25 - 1.5 * iqr)) | 
              (df[columna_seleccionada] > (percentil_75 + 1.5 * iqr))]
porcentaje_atipicos = round((len(outliers) / len(df)) * 100, 2)

st.markdown("<h5>Valores estadisticos</h5>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"Valor medio: {valor_medio}")
with col2:
    st.write(f"Sesgo: {sesgo}")
with col3:
    st.write(f"% de valores atípicos: {porcentaje_atipicos}%")


st.markdown("<h5>Distribuciones</h5>", unsafe_allow_html=True)
analizar_por_ruta = st.checkbox("Mostrar análisis por ruta")

if analizar_por_ruta:
    fig, ax = plt.subplots(figsize=(6.5, 1.5))  # Ajustar el tamaño de la figura
    sns.countplot(y='ruta_marcada', data=df, order=df['ruta_marcada'].value_counts().index, ax=ax, palette=combined_palette)
        
    # Configurar título y etiquetas
    ax.set_title('Distribución de Rutas Marcadas', fontsize=14)
    ax.set_xlabel('Frecuencia', fontsize=12)
    ax.set_ylabel('Ruta Marcada', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

col1, col2 = st.columns(2)

if columna_seleccionada == 'num/vehiculo' and not analizar_por_ruta:
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    sns.histplot(df[columna_seleccionada], kde=True, ax=ax)
    ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}', fontsize=18)
    ax.set_xlabel(columna_seleccionada, fontsize=16)  # Etiqueta del eje x más grande
    ax.set_ylabel('Frecuencia', fontsize=16)
    # Ajustar el tamaño de los ticks
    ax.tick_params(axis='both', labelsize=14)
    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

    # Crear una tabla pivote que cuente los viajes por cada vehículo y día de la semana
    pivot_table = df.pivot_table(index='dia_n', columns='num/vehiculo', aggfunc='size', fill_value=0)

    # Configuración del mapa de calor
    plt.figure(figsize=(20, 6))
    heatmap = sns.heatmap(pivot_table, annot=True, fmt="d", cmap=combined_palette, cbar_kws={'label': 'Cantidad de viajes'})

    # Etiquetas y título
    plt.ylabel('Día de la semana',fontsize=16)
    plt.xlabel('Vehículo',fontsize=16)
    plt.title('Cantidad de viajes por vehículo y día de la semana', fontsize=18)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14, rotation=0)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14, rotation=0)
    # Mostrar el heatmap en Streamlit
    st.pyplot(plt)
elif columna_seleccionada != 'num/vehiculo' and  columna_seleccionada != 'ruta_marcada' and not analizar_por_ruta:
    with col1:    

        fig, ax = plt.subplots()
        sns.histplot(df[columna_seleccionada], kde=True, ax=ax)
        ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}', fontsize=18)
        ax.set_xlabel(columna_seleccionada, fontsize=16)  # Etiqueta del eje x más grande
        ax.set_ylabel('Frecuencia', fontsize=16)
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)
        # Mostrar gráfico en Streamlit
        st.pyplot(fig)

    with col2:

        fig, ax = plt.subplots()
        sns.boxplot(x=df[columna_seleccionada], ax=ax)
        ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}', fontsize=18)
        ax.set_xlabel(columna_seleccionada, fontsize=16)  # Etiqueta del eje x más grande
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)
elif columna_seleccionada == 'num/vehiculo' and  analizar_por_ruta:
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    sns.histplot(data=df, x=columna_seleccionada, kde=True, ax=ax,hue='ruta_marcada', palette=combined_palette)
    ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}', fontsize=18)
    ax.set_xlabel(columna_seleccionada, fontsize=16)  # Etiqueta del eje x más grande
    ax.set_ylabel('Frecuencia', fontsize=16)
    # Ajustar el tamaño de los ticks
    ax.tick_params(axis='both', labelsize=14)
    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

    # Crear una tabla pivote que cuente los viajes por cada vehículo y día de la semana
    pivot_table = df.pivot_table(index='ruta_marcada', columns='num/vehiculo', aggfunc='size', fill_value=0)

    # Configuración del mapa de calor
    plt.figure(figsize=(20, 6))
    heatmap = sns.heatmap(pivot_table, annot=True, fmt="d", cmap=combined_palette, cbar_kws={'label': 'Cantidad de viajes'})

    # Etiquetas y título
    plt.ylabel('Ruta marcada',fontsize=16)
    plt.xlabel('Vehículo',fontsize=16)
    plt.title('Cantidad de viajes por vehículo y ruta', fontsize=18)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14, rotation=0)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14, rotation=0)
    # Mostrar el heatmap en Streamlit
    st.pyplot(plt)
elif columna_seleccionada != 'num/vehiculo' and  columna_seleccionada != 'ruta_marcada' and analizar_por_ruta:
    with col1:    

        fig, ax = plt.subplots()
        sns.histplot(data=df,x=columna_seleccionada, kde=True, ax=ax, hue='ruta_marcada',palette=combined_palette)
        ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}', fontsize=18)
        ax.set_xlabel(columna_seleccionada, fontsize=16)  # Etiqueta del eje x más grande
        ax.set_ylabel('Frecuencia', fontsize=16)
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)
        # Mostrar gráfico en Streamlit
        st.pyplot(fig)

    with col2:

        fig, ax = plt.subplots()
        sns.boxplot(data=df,x='ruta_marcada',y=columna_seleccionada, ax=ax,palette=combined_palette)
        ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}', fontsize=18)
        ax.set_xlabel(columna_seleccionada, fontsize=16)  # Etiqueta del eje x más grande
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)



#estacionalidad
if columna_seleccionada != 'num/vehiculo' and columna_seleccionada != 'ruta_marcada':
    st.markdown("<h5>Gráficos de estacionalidad por ruta</h5>", unsafe_allow_html=True)

    # Calcular el número de apariciones de cada ruta
    ruta_counts = df['ruta_marcada'].value_counts()

    # Identificar las rutas que aparecen solo una vez
    rutas_a_eliminar = ruta_counts[ruta_counts == 1].index

    # Eliminar las filas donde 'ruta_marcada' es una de las rutas a eliminar
    df = df[~df['ruta_marcada'].isin(rutas_a_eliminar)]

    opciones_ruta = df['ruta_marcada'].value_counts().index

    # Selección de columna con 'tipo_negocio' como predeterminado
    ruta_seleccionada = st.selectbox(
        "Selecciona la ruta para graficar:", 
        opciones_ruta,
        key='ruta_seleccionada')

    df_ruta=df[df['ruta_marcada']== ruta_seleccionada]

    # Paso 1: Eliminar índices duplicados, manteniendo la primera ocurrencia
    df_ruta = df_ruta[~df_ruta.index.duplicated(keep='first')]

    # Análisis de la periodicidad del dataset
    df_ruta['df_time_diffs'] = df_ruta.index.to_series().diff().dt.total_seconds()

    # Calcular la mediana de las diferencias de tiempo
    mediana_dif = df_ruta['df_time_diffs'].median()
    # Convertir la mediana a minutos
    mediana_minutos = mediana_dif / 60
    frecuencia = f"{round(mediana_minutos)}T"

    # Cambiar la frecuencia 
    df2 = df_ruta.asfreq(freq=frecuencia, method='bfill')
    
    df2 = df2.rename(columns={'Fecha_Hora_Salida': 'Fecha_Hora'})
    

    col1, col2 = st.columns(2)

    # Agregar contenido en la primera columna
    with col1:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        df_ruta['dia_n'] = df_ruta.index.day_name()
        medianas = df_ruta.groupby('dia_n')[columna_seleccionada].median()
        sns.boxplot(df_ruta, x='dia_n',y=columna_seleccionada, ax=ax, order=medianas.index)
        medianas.plot(style='o-', markersize=8, label='Mediana',lw=0.5, ax=ax)
        ax.set_ylabel(columna_seleccionada, fontsize=16)
        ax.set_xlabel('dia', fontsize=16) 
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)
        st.pyplot(fig)

        # Agregar contenido en la segunda columna
    with col2:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        # Orden para las jornadas
        jornada_order = ['Madrugada', 'Mañana', 'Tarde', 'Noche']

        # Crear el boxplot con transparencia
        sns.boxplot(x='jornada', y=columna_seleccionada, data=df_ruta, ax=ax, order=jornada_order)  # Outliers en rojo y semi-transparentes

        # Añadir la línea de mediana por jornada
        medianas = df_ruta.groupby('jornada',observed=False)[columna_seleccionada].median().reindex(jornada_order)
        ax.plot(jornada_order, medianas, 'o-',  markersize=8, label='Mediana',lw=0.5)  # Mediana como bola azul

        # Etiquetas y título
        ax.set_ylabel(columna_seleccionada, fontsize=16)
        ax.set_xlabel('jornada', fontsize=16) 
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)
        st.pyplot(fig)


    fig, ax = plt.subplots(figsize=(8.5, 3))
    df2['hora'] = df2.index.hour
    medianas = df2.groupby('hora')[columna_seleccionada].median()
    sns.boxplot(df2, x='hora',y=columna_seleccionada, ax=ax, order=medianas.index)
    ax.plot(medianas.index, medianas.values, 'o-',  markersize=8, label='Mediana', lw=0.5)
    ax.set_ylabel(columna_seleccionada, fontsize=12)
    ax.set_xlabel('hora', fontsize=12) 
    # Ajustar el tamaño de los ticks
    ax.tick_params(axis='both', labelsize=10)
    st.pyplot(fig)

#series de tiempo
if columna_seleccionada != 'num/vehiculo':
    st.markdown("<h5>Series de tiempo por ruta</h5>", unsafe_allow_html=True)
    # Crear la figura
    fig = go.Figure()
  
    # Agregar la traza de entrenamiento
    fig.add_trace(go.Scatter(x=df_ruta.index, y=df_ruta[columna_seleccionada], mode='lines', name='Train', line=dict(color="#FFB300") ))

    # Configurar el layout de la figura
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title=columna_seleccionada,
        legend_title="Partición:",
        width=850,
        height=400,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.001,
        )
    )

    # Mostrar el range slider en el eje X
    fig.update_xaxes(rangeslider_visible=True)

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)


    st.subheader('Análisis de la periodicidad del dataset por ruta')

    # Obtener los valores mínimo y máximo de la columna 'df_time_diffs'
    min_val = df_ruta['df_time_diffs'].min()
    max_val = df_ruta['df_time_diffs'].max()

    # Configurar los límites del zoom mediante un slider en Streamlit
    zoom_min, zoom_max = st.slider(
        'Selecciona el rango para hacer zoom en el eje X',
        min_value=float(min_val), max_value=float(max_val),
        value=(float(min_val), float(max_val / 4))
    )

    # Configurar el tamaño de la figura
    fig, ax = plt.subplots(figsize=(6.5, 2))

    # Crear el histograma con KDE
    sns.histplot(df_ruta['df_time_diffs'].dropna(), kde=True, ax=ax)

    # Configurar los límites del eje X basados en el slider
    ax.set_xlim(zoom_min, zoom_max)

    # Asignar nombres a los ejes y el título
    ax.set_xlabel('Diferencia entre observaciones (segundos)')
    ax.set_ylabel('Frecuencia')

    # Agregar una línea vertical en la mediana
    ax.axvline(mediana_dif, color='#004D40', linestyle='--', label='Mediana: {:.2f} s ({:.2f} min)'.format(mediana_dif, mediana_minutos))
    ax.legend()

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    # Mensaje sobre la mediana
    st.write(f"La frecuencia mediana es de {mediana_dif:.2f} segundos, que son {mediana_minutos:.2f} minutos. La vamos a tomar como {round(mediana_minutos)} minutos.")

st.subheader('Modelo de Machine Learning')

# Crear un selector para elegir la columna

#decision tree
st.write('##### RandomForestRegressor')
st.write('El modelo de Random Forest Regressor es un modelo supervisado que utiliza un conjunto de árboles de 
decisión para realizar predicciones continuas. Este método combina múltiples árboles de decisión,
donde cada uno se entrena con un subconjunto diferente de datos y características,
y luego promedia las predicciones para obtener un resultado final más robusto y preciso')

col1, col2, col3 = st.columns(3)
with col1:
    columna_modelo = st.selectbox(
        "Selecciona la variable:",
        ['pasajeros', 'duracion', 'tiempo_terminal'],
        index=0,
        key='columna_modelo_seleccion'
    )



    # Separar variables predictoras (X) y variable objetivo (y)
    X = df[['dia_n', 'hora', 'ruta_marcada']]
    # Supongamos que estos son los codificadores usados durante el entrenamiento
    ruta_encoder = LabelEncoder()
    dia_encoder = LabelEncoder()

    X['ruta_marcada'] = ruta_encoder.fit_transform(X['ruta_marcada'])
    X['dia_n'] = dia_encoder.fit_transform(X['dia_n'])

    y = df[columna_modelo]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Crear el modelo Random Forest
    model1 =  RandomForestRegressor(n_estimators = 60,
                                criterion = 'absolute_error',
                                min_samples_split = 7,
                                min_samples_leaf = 6,
                                max_features = 'sqrt',
                                max_depth= 12
                                )
    model2 = GradientBoostingRegressor(
                                        loss='absolute_error',
                                        max_depth= 4,
                                        learning_rate = 0.12,
                                        random_state=42
                                        )

    model3 = XGBRegressor(objective='reg:absoluteerror', random_state=42)

    model = model1

    model.fit(X_train, y_train)

    # Evaluar el modelo con los mejores parámetros
    y_pred = model.predict(X_test)
    mse = mean_absolute_error(y_test, y_pred)

    # Crear un diccionario con la asignación
    asignaciones_ruta = {clase: idx for idx, clase in enumerate(ruta_encoder.classes_)}
    asignaciones_dia = {clase: idx for idx, clase in enumerate(dia_encoder.classes_)}

with col2:
    # Crear las clases de días de la semana
    dias_semana_spl = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    # Selección del día de la semana por el usuario
    dia_seleccionado = st.selectbox('Selecciona el día de la semana:', dias_semana_spl)
    # Mapeo de días en español a inglés
    dias_map = {
        'Lunes': 'Monday',
        'Martes': 'Tuesday',
        'Miércoles': 'Wednesday',
        'Jueves': 'Thursday',
        'Viernes': 'Friday',
        'Sábado': 'Saturday',
        'Domingo': 'Sunday'
    }
    dia_semana_eng = dias_map[dia_seleccionado]
    # Codificar el día seleccionado
    dia_codificado = asignaciones_dia[dia_semana_eng]

with col3:
    # Crear las clases de días de la semana
    ruta_p = df['ruta_marcada'].value_counts().index
    # Selección del día de la semana por el usuario
    ruta_predecir = st.selectbox('Selecciona la ruta:', ruta_p)
    ruta_codificada = asignaciones_ruta[ruta_predecir]

# Selección de la hora
hora_seleccionada = st.slider('Selecciona la hora del día:', min_value=df['hora'].min(), max_value=df['hora'].max(), value=12)

# Botón para predecir
if st.button('Predecir'):
    # Realizar la predicción
    prediccion = model.predict([[dia_codificado, hora_seleccionada, ruta_codificada]])
    if columna_modelo == 'duracion' or columna_modelo == 'tiempo_terminal':
        total_seconds = prediccion[0] * 60
        hours, remainder = divmod(total_seconds, 3600)  # Obtener horas y el resto
        minutes, seconds = divmod(remainder, 60)  # Obtener minutos y segundos
        
        mensaje = (
            f" La cantidad predicha de {columna_modelo} es: {int(prediccion[0])} minutos. En h, min y s:  {int(hours)}:{int(minutes)}:{seconds:.0f}"
            )
    else:
        mensaje = (
        f" La cantidad predicha de {columna_modelo} es: {int(prediccion[0])} pasajeros"
        )
    
    # Mostrar mensaje con color personalizado
    
    st.markdown(
    f"""
    <style>
        .custom-box {{
            background-color: #FFE5B4;  /* Fondo naranja claro */
            padding: 15px;  /* Espaciado interno */
            border-radius: 10px;  /* Bordes redondeados */
            border: 3px solid #A04000;  /* Borde naranja más oscuro */
            color: #D35400;  /* Texto naranja oscuro */
            font-weight: bold;  /* Texto en negrita */
            text-align: center;  /* Centrar texto horizontalmente */
            font-size: 22px;  /* Tamaño de letra aumentado */
            display: flex;  /* Activar flexbox */
            align-items: center;  /* Centrar verticalmente */
            justify-content: center;  /* Centrar horizontalmente */
            height: 70px;  /* Altura suficiente para centrar */
        }}
    </style>
    <div class="custom-box">
        <b>{mensaje}</b>
    </div>
    """, 
    unsafe_allow_html=True)

st.write(" \n")

# Lógica para formatear el MAE
if columna_modelo == 'duracion' or columna_modelo == 'tiempo_terminal':
    hours, remainder = divmod(mse, 3600)  # Obtener horas y el resto
    minutes, seconds = divmod(remainder, 60)  # Obtener minutos y segundos
        
    mensaje = (
        f" Para este modelo en promedio, la diferencia entre las predicciones del modelo y los valores reales es de {round(mse, 1)} minutos. En h, min y s:  {int(hours)}:{int(minutes)}:{seconds:.0f}"
        )
else:
    mensaje = (
    f" Para este modelo en promedio, la diferencia entre las predicciones del modelo y los valores reales es de {round(mse)} pasajeros por observación"
    )

st.write(mensaje)
