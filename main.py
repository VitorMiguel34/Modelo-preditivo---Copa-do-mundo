
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from normalize_countries import normalize_team_name


try:
    X_treino = pd.read_csv("dados/X_treino.csv")
    Y_treino = pd.Series(pd.read_csv("dados/Y_treino.csv"))
    
    colunas_do_modelo = X_treino.columns.tolist()

    # Instancia o XGBoost Puro com os melhores hiperparâmetros do seu GridSearch
    modelo_copa = XGBClassifier(
        gamma=1, 
        learning_rate=0.05, 
        max_depth=3, 
        n_estimators=400, 
        reg_lambda=10, 
        subsample=1,
        random_state=42 
    )

    modelo_copa.fit(X_treino, Y_treino)
    
except Exception as e:
    raise RuntimeError(f"Erro ao carregar os dados ou treinar o modelo base no main.py: {e}")


def simular_confronto(ano_copa, time_casa, time_fora):
    """
    Prevê o resultado de um jogo e retorna APENAS as probabilidades brutas (0.0 a 100.0).
    Retorna uma tupla: (prob_vitória_casa, prob_empate, prob_vitória_fora)
    Se houver erro (ex: time não encontrado), retorna (0.0, 0.0, 0.0).
    """
    time_casa_norm = normalize_team_name(time_casa)
    time_fora_norm = normalize_team_name(time_fora)
    
    try:
        
        df_ciclo = pd.read_csv(f"dados/dados_processados/estatisticas_ciclos/estatisticas_ciclo{ano_copa}.csv")
        df_ranking = pd.read_csv(f"dados/dados_processados/rankings/fifa_pre_copa/fifa_ranking_pre_copa_{ano_copa}.csv")
        
        df_ciclo["team"] = df_ciclo["team"].apply(normalize_team_name)
        df_ciclo = df_ciclo.set_index("team")
        
        df_ranking["country_full"] = df_ranking["country_full"].apply(normalize_team_name)
        df_ranking = df_ranking.set_index("country_full")
        df_ranking = df_ranking[~df_ranking.index.duplicated(keep='first')]

        
        if time_casa_norm not in df_ciclo.index or time_fora_norm not in df_ciclo.index:
            raise RuntimeError("País não vai para a copa!")

        # Cálculo das features base
        win_rate_casa = df_ciclo.loc[time_casa_norm, "wins"] / df_ciclo.loc[time_casa_norm, "games_played"]
        win_rate_fora = df_ciclo.loc[time_fora_norm, "wins"] / df_ciclo.loc[time_fora_norm, "games_played"]
        
        draw_rate_casa = df_ciclo.loc[time_casa_norm, "draws"] / df_ciclo.loc[time_casa_norm, "games_played"]
        draw_rate_fora = df_ciclo.loc[time_fora_norm, "draws"] / df_ciclo.loc[time_fora_norm, "games_played"]
        
        lose_rate_casa = df_ciclo.loc[time_casa_norm, "loses"] / df_ciclo.loc[time_casa_norm, "games_played"]
        lose_rate_fora = df_ciclo.loc[time_fora_norm, "loses"] / df_ciclo.loc[time_fora_norm, "games_played"]

        rank_casa = df_ranking.loc[time_casa_norm, "rank"] if time_casa_norm in df_ranking.index else 100
        rank_fora = df_ranking.loc[time_fora_norm, "rank"] if time_fora_norm in df_ranking.index else 100

        # Montagem do dicionário de cálculos
        dados_calculados = {
            "wins_difference": win_rate_casa - win_rate_fora,
            "draws_difference": draw_rate_casa - draw_rate_fora,
            "loses_difference": lose_rate_casa - lose_rate_fora,
            "points_per_game_difference": df_ciclo.loc[time_casa_norm, "points_per_game"] - df_ciclo.loc[time_fora_norm, "points_per_game"],
            "ranking_difference": rank_casa - rank_fora,
            "goal_diff_pg_difference": df_ciclo.loc[time_casa_norm, "goal_diff_per_game"] - df_ciclo.loc[time_fora_norm, "goal_diff_per_game"],
            "goal_diff_per_game_difference": df_ciclo.loc[time_casa_norm, "goal_diff_per_game"] - df_ciclo.loc[time_fora_norm, "goal_diff_per_game"]
        }

        # Montar o DataFrame de input filtrando e ordenando exatamente igual ao X de treino
        input_modelo = pd.DataFrame([{col: dados_calculados.get(col, 0) for col in colunas_do_modelo}])
        input_modelo = input_modelo.fillna(0)

        # Previsão das probabilidades com o XGBoost
        probabilidades = modelo_copa.predict_proba(input_modelo)[0]
        
        # Mapeamento seguro das classes (0=Casa, 1=Empate, 2=Fora)
        classes = modelo_copa.classes_
        resultado_dict = dict(zip(classes, probabilidades))
        
        prob_casa = resultado_dict.get(0, resultado_dict.get('Home', 0)) 
        prob_empate = resultado_dict.get(1, resultado_dict.get('Draw', 0)) 
        prob_fora = resultado_dict.get(2, resultado_dict.get('Away', 0)) 

        return float(prob_casa), float(prob_empate), float(prob_fora)
        
    except RuntimeError as e:
        return e