import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# pip install deap
from deap import base, creator, tools, algorithms

# --- Configuración Inicial DEAP ---
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Funciones de Carga y Procesamiento de Datos ---
@st.cache_data
def load_data_and_counts(uploaded_file):
    if uploaded_file is None: return None, {}, {}, {}, [], {}, 0, {}
    try:
        df = pd.read_csv(uploaded_file)
        if 'Numero' not in df.columns or 'Atraso' not in df.columns or 'Frecuencia' not in df.columns:
            st.error("El archivo debe contener las columnas 'Numero', 'Atraso' y 'Frecuencia'."); return None, {}, {}, {}, [], {}, 0, {}
        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce')
        df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce')
        df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce')
        df.dropna(subset=['Numero', 'Atraso', 'Frecuencia'], inplace=True)
        df['Numero'] = df['Numero'].astype(int).astype(str)
        df['Atraso'] = df['Atraso'].astype(int)
        df['Frecuencia'] = df['Frecuencia'].astype(int)
        st.success("Archivo de datos cargado exitosamente.")
        numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
        numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        atrasos_disponibles_int = sorted(df['Atraso'].unique())
        numeros_validos = list(numero_a_atraso.keys())
        distribucion_probabilidad = {num: 1.0/len(numeros_validos) for num in numeros_validos} if numeros_validos else {}
        atraso_counts = df['Atraso'].value_counts().to_dict() 
        atraso_stats = {"min": df['Atraso'].min(), "max": df['Atraso'].max(), "p25": df['Atraso'].quantile(0.25), "p75": df['Atraso'].quantile(0.75)}
        total_atraso_dataset = df['Atraso'].sum()
        return df, numero_a_atraso, numero_a_frecuencia, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset, atraso_stats
    except Exception as e:
        st.error(f"Error al procesar el archivo de datos: {e}"); return None, {}, {}, {}, [], {}, 0, {}

@st.cache_data
def load_historical_combinations(uploaded_file):
    if uploaded_file is None: return []
    try:
        df_hist = pd.read_csv(uploaded_file, header=None)
        historical_sets = [set(pd.to_numeric(row, errors='coerce').dropna().astype(int)) for _, row in df_hist.iterrows()]
        historical_sets = [s for s in historical_sets if len(s) >= 6]
        if historical_sets: st.success(f"Archivo de historial cargado: {len(historical_sets)} combinaciones.")
        else: st.warning("El archivo de historial no contenía combinaciones válidas.")
        return historical_sets
    except Exception as e:
        st.error(f"Error al procesar el archivo de historial: {e}"); return []

# --- Funciones de Análisis Histórico ---
@st.cache_data
def analyze_historical_special_calc(historical_sets, total_atraso_dataset, numero_a_atraso):
    if not historical_sets or total_atraso_dataset is None: return None
    values = [total_atraso_dataset + 40 - sum(numero_a_atraso.get(str(num), 0) for num in s) for s in historical_sets]
    if not values: return None
    return {"min": int(np.min(values)), "max": int(np.max(values)), "mean": np.mean(values), "std": np.std(values)}

@st.cache_data
def analyze_historical_frequency_cv(historical_sets, numero_a_frecuencia):
    if not historical_sets or not numero_a_frecuencia: return None
    cv_values = [np.std(freqs) / np.mean(freqs) for s in historical_sets if (freqs := [numero_a_frecuencia.get(str(num), 0) for num in s]) and np.mean(freqs) > 0]
    if not cv_values: return None
    return {"min": np.min(cv_values), "max": np.max(cv_values), "mean": np.mean(cv_values), "std": np.std(cv_values)}

@st.cache_data
def analyze_historical_delay_cv(historical_sets, numero_a_atraso):
    if not historical_sets or not numero_a_atraso: return None
    cv_values = [np.std(delays) / np.mean(delays) for s in historical_sets if (delays := [numero_a_atraso.get(str(num), 0) for num in s]) and np.mean(delays) > 0]
    if not cv_values: return None
    return {"min": np.min(cv_values), "max": np.max(cv_values), "mean": np.mean(cv_values), "std": np.std(cv_values)}

@st.cache_data
def analyze_historical_structure(historical_sets):
    if not historical_sets: return None, None, None
    sums = [sum(s) for s in historical_sets]
    parity_counts = Counter(sum(1 for num in s if num % 2 == 0) for s in historical_sets)
    consecutive_counts = []
    for s in historical_sets:
        nums = sorted(list(s)); max_consecutive = 0; current_consecutive = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1: current_consecutive += 1
            else: max_consecutive = max(max_consecutive, current_consecutive); current_consecutive = 1
        consecutive_counts.append(max(max_consecutive, current_consecutive))
    sum_stats = {"min": int(np.min(sums)), "max": int(np.max(sums)), "mean": np.mean(sums), "std": np.std(sums)}
    return sum_stats, parity_counts, Counter(consecutive_counts)

# --- Función para puntuar y rankear las combinaciones ---
def score_and_rank_combinations(combinations, num_a_atraso, num_a_freq, total_atraso, atraso_counts_int, historical_stats):
    scored_combinations = []
    means = {key: stats['mean'] for key, stats in historical_stats.items() if stats and 'mean' in stats}
    stds = {key: stats['std'] for key, stats in historical_stats.items() if stats and 'std' in stats}
    
    for combo in combinations:
        atrasos = [num_a_atraso.get(str(n), 0) for n in combo]
        frecuencias = [num_a_freq.get(str(n), 0) for n in combo]
        
        suma = np.sum(combo)
        cv_frec = np.std(frecuencias) / np.mean(frecuencias) if np.mean(frecuencias) > 0 else 0
        cv_atraso = np.std(atrasos) / np.mean(atrasos) if np.mean(atrasos) > 0 else 0
        calc_esp = total_atraso + 40 - sum(atrasos)
        
        score = 1.0
        
        if 'suma' in means and stds.get('suma', 0) > 0:
            score *= np.exp(-0.5 * ((suma - means['suma']) / stds['suma']) ** 2)
        if 'cv_frecuencia' in means and stds.get('cv_frecuencia', 0) > 0:
            score *= np.exp(-0.5 * ((cv_frec - means['cv_frecuencia']) / stds['cv_frecuencia']) ** 2)
        if 'cv_atraso' in means and stds.get('cv_atraso', 0) > 0:
            score *= np.exp(-0.5 * ((cv_atraso - means['cv_atraso']) / stds['cv_atraso']) ** 2)
        if 'calculo_especial' in means and stds.get('calculo_especial', 0) > 0:
            score *= np.exp(-0.5 * ((calc_esp - means['calculo_especial']) / stds['calculo_especial']) ** 2)
            
        scarcity_score = sum(1.0 / atraso_counts_int.get(atr, 1) for atr in atrasos)
        score *= (1 + np.log1p(scarcity_score))

        scored_combinations.append({
            "Puntuación": score,
            "Combinación": " - ".join(map(str, sorted(combo))),
            "CV Frecuencia": cv_frec,
            "CV Atraso": cv_atraso,
            "Cálculo Especial": calc_esp,
            "Suma": suma
        })
        
    return sorted(scored_combinations, key=lambda x: x["Puntuación"], reverse=True)

# --- Motores de Generación y Filtrado (Completados) ---
def generar_combinaciones_con_restricciones(params):
    dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, sim_n_comb, historical_combinations_set, total_atraso, special_calc_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold, delay_cv_range = params
    
    numeros = list(dist_prob.keys())
    probabilidades = list(dist_prob.values())
    valid_combos = []

    for _ in range(sim_n_comb):
        combo = np.random.choice(numeros, size=n_selecciones, replace=False, p=probabilidades)
        combo = [int(x) for x in combo]
        
        # Filtro Suma
        suma = sum(combo)
        if not (sum_range[0] <= suma <= sum_range[1]): continue
            
        # Filtro CV Atraso
        atrasos = [num_a_atraso.get(str(n), 0) for n in combo]
        cv_atraso = np.std(atrasos) / np.mean(atrasos) if np.mean(atrasos) > 0 else 0
        if not (delay_cv_range[0] <= cv_atraso <= delay_cv_range[1]): continue
            
        valid_combos.append((combo, 0)) # 0 es un placeholder de score
    
    return valid_combos

def procesar_combinaciones(params_tuple, n_ejec):
    resultados = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generar_combinaciones_con_restricciones, params_tuple) for _ in range(n_ejec)]
        for future in as_completed(futures):
            resultados.append(future.result())
    return resultados

def filtrar_por_composicion(combinaciones, numero_a_atraso, composicion_rules):
    filtradas = []
    # Rangos por defecto para los grupos (puedes ajustar esto según tu necesidad)
    rango_caliente = (0, 4)
    rango_tibio = (5, 9)
    rango_frio = (10, 19)
    
    for combo in combinaciones:
        calientes = tibios = frios = congelados = 0
        for num in combo:
            atr = numero_a_atraso.get(str(num), -1)
            if rango_caliente[0] <= atr <= rango_caliente[1]: calientes += 1
            elif rango_tibio[0] <= atr <= rango_tibio[1]: tibios += 1
            elif rango_frio[0] <= atr <= rango_frio[1]: frios += 1
            elif atr >= 20: congelados += 1
            
        if (calientes == composicion_rules.get('Calientes', 0) and
            tibios == composicion_rules.get('Tibios', 0) and
            frios == composicion_rules.get('Fríos', 0) and
            congelados == composicion_rules.get('Congelados', 0)):
            filtradas.append(combo)
    return filtradas

def evaluar_individuo_deap(individuo, params):
    dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, historical_combinations_set, total_atraso, special_calc_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold, delay_cv_range = params
    
    if len(set(individuo)) != len(individuo): return -9999.0, # Penalizar repeticiones
    
    suma = sum(individuo)
    if not (sum_range[0] <= suma <= sum_range[1]): return -100.0,
    
    atrasos = [num_a_atraso.get(str(n), 0) for n in individuo]
    cv_atraso = np.std(atrasos) / np.mean(atrasos) if np.mean(atrasos) > 0 else 0
    
    # Fitness simple basado en estar dentro del rango de CV de atraso
    if delay_cv_range[0] <= cv_atraso <= delay_cv_range[1]:
        return 100.0 - abs(cv_atraso - ((delay_cv_range[0]+delay_cv_range[1])/2)),
    return -50.0,

def ejecutar_algoritmo_genetico(ga_params, backend_params):
    ga_ngen, ga_npob, ga_cxpb, ga_mutpb, dist_prob, n_selecciones = ga_params
    numeros_validos = list(dist_prob.keys())
    
    toolbox = base.Toolbox()
    toolbox.register("attr_item", random.choice, numeros_validos)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, n_selecciones)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluar_individuo_deap, params=backend_params)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=int(min(numeros_validos)), up=int(max(numeros_validos)), indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=ga_npob)
    hof = tools.HallOfFame(1)
    
    try:
        algorithms.eaSimple(pop, toolbox, cxpb=ga_cxpb, mutpb=ga_mutpb, ngen=ga_ngen, halloffame=hof, verbose=False)
        return hof[0], pop, None
    except Exception as e:
        return None, None, str(e)


# ----------------------- Interfaz Gráfica de Streamlit -----------------------
st.set_page_config(layout="wide", page_title="Generador de Combinaciones de Precisión")
st.title("Modelo Homeostático de Precisión")

# --- INICIALIZACIÓN DE VARIABLES PARA EVITAR EL NameError ---
df = None
num_a_atraso = {}
num_a_freq = {}
dist_prob = {}
atraso_counts = {}
total_atraso = 0
historical_combinations_set = []

# --- 1. CARGA DE DATOS ---
st.sidebar.header("1. Carga de Archivos")
file_datos = st.sidebar.file_uploader("Sube tu archivo de Datos (CSV)", type=["csv"])
file_hist = st.sidebar.file_uploader("Sube tu archivo Histórico (CSV)", type=["csv"])

if file_datos:
    df, num_a_atraso, num_a_freq, dist_prob, atrasos_disponibles_int, atraso_counts, total_atraso, atraso_stats = load_data_and_counts(file_datos)

if file_hist:
    historical_combinations_set = load_historical_combinations(file_hist)

# --- 2. PARÁMETROS Y FILTROS ---
st.sidebar.header("2. Parámetros Principales")
n_selecciones = st.sidebar.number_input("Tamaño de la combinación (ej. 6)", min_value=3, max_value=20, value=6)

with st.expander("⚙️ Parámetros de Simulación en Cascada", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        sim_n_comb = st.number_input("Combinaciones por ejecución", min_value=1000, max_value=50000, value=5000)
    with col2:
        sim_n_ejec = st.number_input("Número de ejecuciones", min_value=1, max_value=20, value=5)
        
with st.expander("🧬 Parámetros del Algoritmo Genético", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    ga_ngen = c1.number_input("Generaciones", value=50)
    ga_npob = c2.number_input("Población", value=100)
    ga_cxpb = c3.slider("Prob. Cruce", 0.1, 1.0, 0.7)
    ga_mutpb = c4.slider("Prob. Mutación", 0.01, 0.5, 0.2)

with st.expander("📊 Filtros Homeostáticos", expanded=False):
    sum_range = st.slider("Rango de Suma Permitido", 10, 300, (100, 200))
    delay_cv_range = st.slider("Rango de CV de Atraso", 0.0, 3.0, (0.5, 1.5))
    freq_cv_range = (0, 3.0) # Dummy si no hay UI
    special_calc_range = (0, 1000) # Dummy
    parity_counts_allowed = [2, 3, 4] # Dummy
    max_consecutive_allowed = 2 # Dummy
    hist_similarity_threshold = 4 # Dummy
    restricciones_finales = {}

with st.expander("🧩 Filtro de Composición (Etapa 2)", expanded=True):
    st.write("Selecciona cuántos números deseas de cada grupo:")
    c1, c2, c3, c4 = st.columns(4)
    comp_calientes = c1.number_input("Calientes", 0, n_selecciones, 2)
    comp_tibios = c2.number_input("Tibios", 0, n_selecciones, 2)
    comp_frios = c3.number_input("Fríos", 0, n_selecciones, 1)
    comp_congelados = c4.number_input("Congelados", 0, n_selecciones, 1)
    
    composicion_rules = {'Calientes': comp_calientes, 'Tibios': comp_tibios, 'Fríos': comp_frios, 'Congelados': comp_congelados}
    total_count_composition = sum(composicion_rules.values())

# --- 3. SECCIÓN DE EJECUCIÓN CORREGIDA Y MEJORADA ---
st.header("3. Ejecutar Algoritmos")

# AQUÍ ES DONDE OCURRÍA EL ERROR. AHORA ESTÁ PROTEGIDO Y LAS VARIABLES EXISTEN.
if df is not None:
    backend_params = (dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, sim_n_comb, historical_combinations_set, total_atraso, special_calc_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold, delay_cv_range)
    
    run_col1, run_col2 = st.columns(2)

    with run_col1:
        if st.button("Ejecutar Algoritmo Genético"):
            with st.spinner("Buscando la mejor combinación..."):
                ga_params = (ga_ngen, ga_npob, ga_cxpb, ga_mutpb, dist_prob, n_selecciones)
                mejor_ind, _, err_msg = ejecutar_algoritmo_genetico(ga_params, backend_params)
            if err_msg: 
                st.error(err_msg)
            elif mejor_ind:
                st.subheader("Mejor Combinación (GA)")
                st.success(f"**Combinación: {' - '.join(map(str, sorted(mejor_ind)))}**")
                freqs = [num_a_freq.get(str(v),0) for v in mejor_ind]
                delays = [num_a_atraso.get(str(v),0) for v in mejor_ind]
                st.write(f"**CV Frecuencia:** {np.std(freqs)/np.mean(freqs) if np.mean(freqs) > 0 else 0:.2f}")
                st.write(f"**CV Atraso:** {np.std(delays)/np.mean(delays) if np.mean(delays) > 0 else 0:.2f}")
                st.write(f"**Cálculo Especial:** {total_atraso + 40 - sum(delays)}")
            else: 
                st.warning("El GA no encontró una combinación válida.")

    with run_col2:
        if st.button("Ejecutar Simulación en Cascada"):
            params_sim = backend_params
            with st.spinner("Etapa 1: Generando combinaciones..."):
                start_time = time.time()
                resultados = procesar_combinaciones(params_sim, sim_n_ejec)
                st.info(f"Etapa 1: {sum(len(r) for r in resultados)} combinaciones generadas en {time.time() - start_time:.2f} s.")
            
            todas_unicas = list(set(tuple(int(n) for n in c) for res in resultados for c, _ in res))
            st.info(f"**{len(todas_unicas)}** combinaciones únicas válidas tras el filtro primario.")
            
            combinaciones_a_rankear = todas_unicas
            
            if total_count_composition == n_selecciones:
                with st.spinner("Etapa 2: Aplicando filtro de composición..."):
                    combinaciones_filtradas = filtrar_por_composicion(todas_unicas, num_a_atraso, composicion_rules)
                st.success(f"Etapa 2: **{len(combinaciones_filtradas)}** combinaciones cumplen el perfil de composición.")
                combinaciones_a_rankear = combinaciones_filtradas
            else:
                st.warning(f"Filtro de composición ignorado (La suma de los grupos es {total_count_composition}, pero debe ser exactamente {n_selecciones}).")
            
            if combinaciones_a_rankear:
                with st.spinner("Etapa 3: Puntuando y rankeando las mejores combinaciones..."):
                    sum_stats, _, _ = analyze_historical_structure(historical_combinations_set)
                    freq_cv_stats = analyze_historical_frequency_cv(historical_combinations_set, num_a_freq)
                    delay_cv_stats = analyze_historical_delay_cv(historical_combinations_set, num_a_atraso)
                    special_calc_stats = analyze_historical_special_calc(historical_combinations_set, total_atraso, num_a_atraso)
                    
                    historical_stats = {
                        'suma': sum_stats,
                        'cv_frecuencia': freq_cv_stats,
                        'cv_atraso': delay_cv_stats,
                        'calculo_especial': special_calc_stats
                    }
                    
                    # Convertir keys de string a int para la función de puntuación
                    atraso_counts_int = {int(k): v for k, v in atraso_counts.items() if str(k).isdigit() or isinstance(k, int)}

                    ranked_results = score_and_rank_combinations(
                        combinaciones_a_rankear, 
                        num_a_atraso, 
                        num_a_freq, 
                        total_atraso,
                        atraso_counts_int,
                        historical_stats
                    )
                
                st.subheader(f"🏆 Top Combinaciones Más Potentes ({len(ranked_results)})")
                df_results = pd.DataFrame(ranked_results)
                
                # Reordenar y formatear columnas para mejor visualización
                cols_to_show = ["Puntuación", "Combinación", "Suma", "CV Atraso", "CV Frecuencia", "Cálculo Especial"]
                df_results = df_results[cols_to_show]
                
                df_results['Puntuación'] = df_results['Puntuación'].map('{:,.4f}'.format)
                df_results['CV Frecuencia'] = df_results['CV Frecuencia'].map('{:,.2f}'.format)
                df_results['CV Atraso'] = df_results['CV Atraso'].map('{:,.2f}'.format)

                st.dataframe(df_results)
            else:
                st.error("No quedaron combinaciones después de aplicar los filtros.")
else:
    st.warning("⚠️ Sube primero tu archivo de Datos CSV en la barra lateral para poder ejecutar los algoritmos.")

# --- Guía Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("Guía del Modelo de 3 Etapas")
st.sidebar.markdown("""
Este modelo utiliza un enfoque sofisticado para identificar las combinaciones con mayor potencial.

**Etapa 1: Generación y Filtrado**
- Se crean combinaciones aleatorias.
- Se aplica un conjunto de **filtros homeostáticos y estructurales** para descartar lo improbable.

**Etapa 2: Filtrado Estratégico (Opcional)**
- Si defines un perfil de **Composición** exacto, se aplica este filtro para refinar a una estrategia específica.

**Etapa 3: Puntuación y Ranking**
- A cada combinación se le asigna una **"Puntuación de Potencia"** basada en cercanía al histórico y la escasez de los números.
""")
