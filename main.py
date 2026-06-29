import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from normalize_countries import normalize_team_name

# ==============================================================================
# MULTIPLICADOR DE FORÇA REALISTA (Ajustado para impor o favoritismo da Espanha)
# ==============================================================================
FORCA_SELECAO = {
    # TIER 1A - O Favoritismo Histórico e Técnico Absoluto
    'Spain': 1.55,       # Elevada ao topo máximo para esmagar qualquer zebra no grupo e mata-mata
    'Argentina': 1.4, 'Brazil': 1.4, 'France': 1.4, 'England': 1.4, 
    
    # TIER 1B - Grandes Potências
    'Portugal': 1.3, 'Netherlands': 1.3, 'Germany': 1.3,
    
    # TIER 2 - Competitivos Avançados (Uruguai ajustado para a realidade histórica)
    'Uruguay': 1.05,     # Reduzido para garantir que a Espanha passe em 1º com folga absoluta
    'Colombia': 1.2, 'Croatia': 1.2, 'Belgium': 1.2, 
    'Italy': 1.1, 'Switzerland': 1.1, 'Morocco': 1.1, 'Senegal': 1.1,
    
    # TIER 3 - Médios / Força Regional Real
    'United States': 1.0, 'Mexico': 1.0, 'Japan': 0.95, 'South Korea': 0.95,
    'Turkey': 0.90, 'Ecuador': 0.95, 'Austria': 0.95, 'Norway': 0.95, 'Czechia': 0.95,
    
    # TIER 4 - Contenção de Zebras Continentais Infladas
    'Iran': 0.70, 'Saudi Arabia': 0.65, 'Qatar': 0.60, 'Iraq': 0.60,
    'Panama': 0.60, 'Canada': 0.75, 'Haiti': 0.50, 'Curacao': 0.50, 'New Zealand': 0.45
}

try:
    X_cru = pd.read_csv("dados/X_treino.csv")
    if "result" in X_cru.columns:
        X_treino = X_cru.drop(columns=["result"])
    else:
        X_treino = X_cru
        
    Y_treino = pd.read_csv("dados/Y_treino.csv")["result"]
    colunas_do_modelo = X_treino.columns.tolist()

    modelo_copa = XGBClassifier(
        gamma=0.1,            
        learning_rate=0.05, 
        max_depth=4,          
        n_estimators=300, 
        reg_lambda=1.0,       
        subsample=0.8,
        random_state=42 
    )
    modelo_copa.fit(X_treino, Y_treino)
    print(f"\n[SUCESSO] Modelo inicializado. Hierarquia do futebol reestabelecida.")
    
except Exception as e:
    raise RuntimeError(f"Erro crítico ao inicializar o modelo no main.py: {e}")


def simular_confronto(ano_copa, time_casa, time_fora):
    time_casa_norm = normalize_team_name(time_casa)
    time_fora_norm = normalize_team_name(time_fora)
    
    try:
        df_ciclo = pd.read_csv(f"dados/estatisticas_ciclos/estatisticas_ciclo{ano_copa}.csv")
        df_ranking = pd.read_csv(f"dados/rankings/fifa_pre_copa/fifa_ranking_pre_copa_{ano_copa}.csv")
        
        df_ciclo["team"] = df_ciclo["team"].apply(normalize_team_name)
        df_ciclo = df_ciclo.set_index("team")
        
        df_ranking.columns = df_ranking.columns.str.lower()
        df_ranking["country_full"] = df_ranking["country_full"].apply(normalize_team_name)
        df_ranking = df_ranking.set_index("country_full")
        df_ranking = df_ranking[~df_ranking.index.duplicated(keep='first')]

        if time_casa_norm not in df_ciclo.index or time_fora_norm not in df_ciclo.index:
            return 0.33, 0.34, 0.33

        col_saldo = "goal_diff_per_game" if "goal_diff_per_game" in df_ciclo.columns else "goal_diff_pg"

        win_rate_casa = df_ciclo.loc[time_casa_norm, "wins"] / df_ciclo.loc[time_casa_norm, "games_played"]
        win_rate_fora = df_ciclo.loc[time_fora_norm, "wins"] / df_ciclo.loc[time_fora_norm, "games_played"]
        
        draw_rate_casa = df_ciclo.loc[time_casa_norm, "draws"] / df_ciclo.loc[time_casa_norm, "games_played"]
        draw_rate_fora = df_ciclo.loc[time_fora_norm, "draws"] / df_ciclo.loc[time_fora_norm, "games_played"]
        
        lose_rate_casa = df_ciclo.loc[time_casa_norm, "loses"] / df_ciclo.loc[time_casa_norm, "games_played"]
        lose_rate_fora = df_ciclo.loc[time_fora_norm, "loses"] / df_ciclo.loc[time_fora_norm, "games_played"]

        saldo_casa = float(df_ciclo.loc[time_casa_norm, col_saldo])
        saldo_fora = float(df_ciclo.loc[time_fora_norm, col_saldo])
        
        ppg_casa = float(df_ciclo.loc[time_casa_norm, "points_per_game"])
        ppg_fora = float(df_ciclo.loc[time_fora_norm, "points_per_game"])

        rk_casa = float(df_ranking.loc[time_casa_norm, "rank"]) if time_casa_norm in df_ranking.index else 100.0
        rk_fora = float(df_ranking.loc[time_fora_norm, "rank"]) if time_fora_norm in df_ranking.index else 100.0
        ranking_diff_val = rk_fora - rk_casa

        dados_calculados = {
            "wins_difference": float(win_rate_casa - win_rate_fora),
            "draws_difference": float(draw_rate_casa - draw_rate_fora),
            "loses_difference": float(lose_rate_casa - lose_rate_fora),
            "points_per_game_difference": float(ppg_casa - ppg_fora),
            "goal_diff_pg_difference": float(saldo_casa - saldo_fora),
            "ranking_difference": float(ranking_diff_val)
        }

        input_list = [dados_calculados.get(col, 0.0) for col in colunas_do_modelo]
        input_modelo = pd.DataFrame([input_list], columns=colunas_do_modelo)

        probabilidades = modelo_copa.predict_proba(input_modelo)[0]
        resultado_dict = dict(zip(modelo_copa.classes_, probabilidades))
        
        prob_casa = resultado_dict.get(2, resultado_dict.get('Home', 0.33)) 
        prob_empate = resultado_dict.get(1, resultado_dict.get('Draw', 0.34)) 
        prob_fora = resultado_dict.get(0, resultado_dict.get('Away', 0.33)) 

        # Aplicação dos multiplicadores reais de peso de camisa/histórico
        mod_casa = FORCA_SELECAO.get(time_casa, 0.85)
        mod_fora = FORCA_SELECAO.get(time_fora, 0.85)
        
        prob_casa = prob_casa * mod_casa
        prob_fora = prob_fora * mod_fora
        
        soma_ajustada = prob_casa + prob_empate + prob_fora
        if soma_ajustada > 0:
            prob_casa /= soma_ajustada
            prob_empate /= soma_ajustada
            prob_fora /= soma_ajustada

        return float(prob_casa), float(prob_empate), float(prob_fora)
        
    except Exception:
        return 0.33, 0.34, 0.33