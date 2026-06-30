import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from normalize_countries import normalize_team_name

# Inicialização e treinamento do modelo puro (sem vazamento de dados)
try:
    X_cru = pd.read_csv("dados/X_treino.csv")
    
    # Prevenção estrita de Data Leakage (Vazamento de Dados)
    if "result" in X_cru.columns:
        X_treino = X_cru.drop(columns=["result"])
    else:
        X_treino = X_cru
        
    Y_treino = pd.read_csv("dados/Y_treino.csv")["result"]
    colunas_do_modelo = X_treino.columns.tolist()

    # Hiperparâmetros calibrados para regularização robusta
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
    print(f"\n[SUCESSO] XGBoost treinado puramente com {len(colunas_do_modelo)} características matemáticas.")
    
except Exception as e:
    raise RuntimeError(f"Erro crítico ao inicializar o modelo no main.py: {e}")



# Função de inferência preditiva (máxima confiabilidade científica)

def simular_confronto(ano_copa, time_casa, time_fora):
    """
    Simula um confronto direto entre duas seleções utilizando estritamente
    a inteligência preditiva do modelo XGBoost baseado no ciclo e Ranking FIFA.
    """
    time_casa_norm = normalize_team_name(time_casa)
    time_fora_norm = normalize_team_name(time_fora)
    
    try:
        # Carrega as tabelas consolidadas pelo pipeline dados.ipynb
        df_ciclo = pd.read_csv(f"dados/estatisticas_ciclos/estatisticas_ciclo{ano_copa}.csv")
        df_ranking = pd.read_csv(f"dados/rankings/fifa_pre_copa/fifa_ranking_pre_copa_{ano_copa}.csv")
        
        # Garante a normalização textual para indexação perfeita
        df_ciclo["team"] = df_ciclo["team"].apply(normalize_team_name)
        df_ciclo = df_ciclo.set_index("team")
        
        df_ranking.columns = df_ranking.columns.str.lower()
        df_ranking["country_full"] = df_ranking["country_full"].apply(normalize_team_name)
        df_ranking = df_ranking.set_index("country_full")
        df_ranking = df_ranking[~df_ranking.index.duplicated(keep='first')]

        # Fallback de segurança para seleções não catalogadas
        if time_casa_norm not in df_ciclo.index or time_fora_norm not in df_ciclo.index:
            return 0.33, 0.34, 0.33

        col_saldo = "goal_diff_per_game" if "goal_diff_per_game" in df_ciclo.columns else "goal_diff_pg"

        # 1. Recuperação das estatísticas puras do ciclo de cada seleção
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

        # Cálculo da métrica de nivelamento internacional (Diferença do Ranking FIFA)
        rk_casa = float(df_ranking.loc[time_casa_norm, "rank"]) if time_casa_norm in df_ranking.index else 100.0
        rk_fora = float(df_ranking.loc[time_fora_norm, "rank"]) if time_fora_norm in df_ranking.index else 100.0
        ranking_diff_val = rk_fora - rk_casa

        # 2. Montagem do vetor de características (Features) estruturado exatamente igual ao treino
        dados_calculados = {
            "wins_difference": float(win_rate_casa - win_rate_fora),
            "draws_difference": float(draw_rate_casa - draw_rate_fora),
            "loses_difference": float(lose_rate_casa - lose_rate_fora),
            "points_per_game_difference": float(ppg_casa - ppg_fora),
            "goal_diff_pg_difference": float(saldo_casa - saldo_fora),
            "ranking_difference": float(ranking_diff_val)
        }

        # Ordenação estrita das colunas conforme mapeamento original do modelo
        input_list = [dados_calculados.get(col, 0.0) for col in colunas_do_modelo]
        input_modelo = pd.DataFrame([input_list], columns=colunas_do_modelo)

        # 3. Predição Pura do Modelo (Livre de qualquer manipulação humana)
        probabilidades = modelo_copa.predict_proba(input_modelo)[0]
        resultado_dict = dict(zip(modelo_copa.classes_, probabilidades))
        
        prob_casa = resultado_dict.get(2, resultado_dict.get('Home', 0.33)) 
        prob_empate = resultado_dict.get(1, resultado_dict.get('Draw', 0.34)) 
        prob_fora = resultado_dict.get(0, resultado_dict.get('Away', 0.33)) 

        # Retorna o vetor puro, cuja soma nativa do XGBoost fecha em 1.0 (100%)
        return float(prob_casa), float(prob_empate), float(prob_fora)
        
    except Exception:
        # Fallback resiliente para garantir a continuidade da simulação em caso de exceções
        return 0.33, 0.34, 0.33