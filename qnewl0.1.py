import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Necesitas instalar deap si no lo tienes: pip install deap
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
        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce'); df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce'); df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce')
        df.dropna(subset=['Numero', 'Atraso', 'Frecuencia'], inplace=True)
        df['Numero'], df['Atraso'], df['Frecuencia'] = df['Numero'].astype(int).astype(str), df['Atraso'].astype(int), df['Frecuencia'].astype(int)
        st.success("Archivo de datos cargado exitosamente.")
        numero_a_atraso = dict(zip(df['Numero'], df['Atraso'])); numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        atrasos_disponibles_int = sorted(df['Atraso'].unique()); numeros_validos = list(numero_a_atraso.keys())
        distribucion_probabilidad = {num: 1.0/len(numeros_validos) for num in numeros_validos} if numeros_validos else {}
        atraso_counts = df['Atraso'].value_counts().to_dict() # Se mantiene str como key para la UI
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
    sums = [sum(s) for s in historical_sets]; parity_counts = Counter(sum(1 for num in s if num % 2 == 0) for s in historical_sets)
    consecutive_counts = []
    for s in historical_sets:
        nums = sorted(list(s)); max_consecutive = 0; current_consecutive = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1: current_consecutive += 1
            else: max_consecutive = max(max_consecutive, current_consecutive); current_consecutive = 1
        consecutive_counts.append(max(max_consecutive, current_consecutive))
    sum_stats = {"min": int(np.min(sums)), "max": int(np.max(sums)), "mean": np.mean(sums), "std": np.std(sums)}
    return sum_stats, parity_counts, Counter(consecutive_counts)

@st.cache_data
def analyze_historical_composition(historical_sets, numero_a_atraso, composicion_ranges):
    if not historical_sets: return None
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    counts = Counter(tuple(Counter(get_category(numero_a_atraso.get(str(num), -1), composicion_ranges) for num in s).get(cat, 0) for cat in ['caliente', 'tibio', 'frio', 'congelado']) for s in historical_sets)
    return counts if counts else None
    
# --- NUEVO: Función para puntuar y rankear las combinaciones ---
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


# --- Motores de Generación y Filtrado ---
def generar_combinaciones_con_restricciones(params):
    # ... (código sin cambios)
    pass

def procesar_combinaciones(params_tuple, n_ejec):
    # ... (código sin cambios)
    pass

def filtrar_por_composicion(combinaciones, numero_a_atraso, composicion_rules):
    # ... (código sin cambios)
    pass

def evaluar_individuo_deap(individuo_str, params):
    # ... (código sin cambios)
    pass

def ejecutar_algoritmo_genetico(ga_params, backend_params):
    # ... (código sin cambios)
    pass

# (Pega aquí las definiciones completas de las funciones de arriba que omití por brevedad)
# ...

# ----------------------- Interfaz Gráfica de Streamlit -----------------------
st.set_page_config(layout="wide", page_title="Generador de Combinaciones de Precisión")
st.title("Modelo Homeostático de Precisión")
if 'suggested_composition' not in st.session_state: st.session_state.suggested_composition = None

# ... (Todo tu código de UI para Cargar Archivos y Filtros va aquí)
# Por favor, asegúrate de pegar toda esa sección aquí.
# ...

# --- SECCIÓN DE EJECUCIÓN CORREGIDA Y MEJORADA ---
st.header("3. Ejecutar Algoritmos")
if df is not None:
    backend_params = (dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, historical_combinations_set, total_atraso, special_calc_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold, delay_cv_range)
    
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
                st.success(f"**Combinación: {' - '.join(map(str, mejor_ind))}**")
                freqs = [num_a_freq.get(str(v),0) for v in mejor_ind]
                delays = [num_a_atraso.get(str(v),0) for v in mejor_ind]
                st.write(f"**CV Frecuencia:** {np.std(freqs)/np.mean(freqs) if np.mean(freqs) > 0 else 0:.2f}")
                st.write(f"**CV Atraso:** {np.std(delays)/np.mean(delays) if np.mean(delays) > 0 else 0:.2f}")
                st.write(f"**Cálculo Especial:** {total_atraso + 40 - sum(delays)}")
            else: 
                st.warning("El GA no encontró una combinación válida.")

    with run_col2:
        if st.button("Ejecutar Simulación en Cascada"):
            params_sim = (dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, sim_n_comb, historical_combinations_set, total_atraso, special_calc_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold, delay_cv_range)
            with st.spinner("Etapa 1: Generando combinaciones..."):
                start_time = time.time()
                resultados = procesar_combinaciones(params_sim, sim_n_ejec)
                st.info(f"Etapa 1: {sum(len(r) for r in resultados)} combinaciones válidas en {time.time() - start_time:.2f} s.")
            
            todas_unicas = list(set(tuple(int(n) for n in c) for res in resultados for c, _ in res))
            st.info(f"**{len(todas_unicas)}** combinaciones únicas generadas.")
            
            combinaciones_a_rankear = todas_unicas
            if total_count_composition == n_selecciones:
                with st.spinner("Etapa 2: Aplicando filtro de composición..."):
                    combinaciones_filtradas = filtrar_por_composicion(todas_unicas, num_a_atraso, composicion_rules)
                st.success(f"Etapa 2: **{len(combinaciones_filtradas)}** combinaciones cumplen el perfil.")
                combinaciones_a_rankear = combinaciones_filtradas
            
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
                    atraso_counts_int = {int(k): v for k, v in atraso_counts.items() if k.isdigit()}

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
                st.warning("No quedaron combinaciones después de aplicar los filtros.")
else:
    st.warning("Carga los archivos de datos para ejecutar los algoritmos.")

# ... (código del sidebar) ...
st.sidebar.header("Guía del Modelo de 3 Etapas")

st.sidebar.markdown("""
Este modelo utiliza un enfoque sofisticado para identificar las combinaciones con mayor potencial.

**Etapa 1: Generación y Filtrado**
- Se crean millones de combinaciones aleatorias.
- Se aplica un conjunto de **filtros homeostáticos y estructurales** (basados en el historial) para descartar el 99.9% de las combinaciones estadísticamente improbables.
- El resultado es un "caldo de cultivo" de miles de candidatos de alta calidad.

**Etapa 2: Filtrado Estratégico (Opcional)**
- Si defines un perfil de **Composición** (ej. 2 Calientes, 3 Tibios, 1 Frío), se aplica este filtro para refinar aún más el "caldo de cultivo" a una estrategia específica.

**Etapa 3: Puntuación y Ranking**
- A cada combinación finalista se le asigna una **"Puntuación de Potencia"**.
- Esta puntuación mide qué tan "perfecta" es una combinación, considerando:
    - **Proximidad al Ideal:** Qué tan cerca están sus métricas de los promedios históricos.
    - **Índice de Escasez:** Un bono si está formada por números de grupos de atraso poco comunes.
- Las combinaciones se presentan en un **ranking ordenado por su Puntuación de Potencia.**
""")
