import os
import pandas as pd
from collections import defaultdict, deque

# ----------------------------------------
# Cargar y limpiar datos
# ----------------------------------------
filename = "atp_matches_1968_2024_completo.csv"
folder_path = '/Users/Stikets/Desktop/Axel/python/Data'
full_path = os.path.join(folder_path, filename)

df = pd.read_csv(full_path)

columnas_utiles = [
    'tourney_date', 'surface', 'round', 'best_of', 'tourney_level',
    'winner_name', 'winner_hand', 'winner_age', 'winner_rank',
    'loser_name', 'loser_hand', 'loser_age', 'loser_rank',
    'w_ace', 'w_df', 'w_1stWon', 'w_2ndWon',
    'l_ace', 'l_df', 'l_1stWon', 'l_2ndWon'
]
df = df[columnas_utiles].copy()

# Limpieza básica
df = df.dropna(subset=['winner_name', 'loser_name', 'surface', 'round'])
valores_numericos = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age',
                     'w_ace', 'w_df', 'w_1stWon', 'w_2ndWon',
                     'l_ace', 'l_df', 'l_1stWon', 'l_2ndWon']
df[valores_numericos] = df[valores_numericos].fillna(0)
df['winner_hand'] = df['winner_hand'].fillna('R')
df['loser_hand'] = df['loser_hand'].fillna('R')

# Ordenar por fecha para cálculos cronológicos
df = df.sort_values("tourney_date").reset_index(drop=True)

# ----------------------------------------
# Calcular ELO
# ----------------------------------------
elo_dict = defaultdict(lambda: 1500)
elo_winner_list = []
elo_loser_list = []

def calcular_elo(elo_w, elo_l, k=32):
    expected_win = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
    return elo_w + k * (1 - expected_win), elo_l - k * (1 - expected_win)

for _, row in df.iterrows():
    w, l = row['winner_name'], row['loser_name']
    ew, el = elo_dict[w], elo_dict[l]
    elo_winner_list.append(ew)
    elo_loser_list.append(el)
    elo_dict[w], elo_dict[l] = calcular_elo(ew, el)

df['elo_winner'] = elo_winner_list
df['elo_loser'] = elo_loser_list

# ----------------------------------------
# Head to Head
# ----------------------------------------
h2h_counter = defaultdict(int)
h2h_winner_vs_loser = []
h2h_loser_vs_winner = []

for _, row in df.iterrows():
    w, l = row['winner_name'], row['loser_name']
    h2h_winner_vs_loser.append(h2h_counter[(w, l)])
    h2h_loser_vs_winner.append(h2h_counter[(l, w)])
    h2h_counter[(w, l)] += 1

df['h2h_winner_vs_loser'] = h2h_winner_vs_loser
df['h2h_loser_vs_winner'] = h2h_loser_vs_winner

# ----------------------------------------
# Racha reciente
# ----------------------------------------
N = 5
recent_results = defaultdict(lambda: deque(maxlen=N))
recent_winner_wins = []
recent_loser_wins = []

for _, row in df.iterrows():
    w, l = row['winner_name'], row['loser_name']
    recent_winner_wins.append(sum(recent_results[w]))
    recent_loser_wins.append(sum(recent_results[l]))
    recent_results[w].append(1)
    recent_results[l].append(0)

df['recent_winner_wins'] = recent_winner_wins
df['recent_loser_wins'] = recent_loser_wins

# ----------------------------------------
# Superficie favorita
# ----------------------------------------
surface_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
surface_winner_wr = []
surface_loser_wr = []

for _, row in df.iterrows():
    s, w, l = row['surface'], row['winner_name'], row['loser_name']
    def winrate(player):
        stats = surface_stats[(player, s)]
        return stats['wins'] / stats['total'] if stats['total'] > 0 else 0.5
    surface_winner_wr.append(winrate(w))
    surface_loser_wr.append(winrate(l))
    surface_stats[(w, s)]['wins'] += 1
    surface_stats[(w, s)]['total'] += 1
    surface_stats[(l, s)]['total'] += 1

df['surface_winner_wr'] = surface_winner_wr
df['surface_loser_wr'] = surface_loser_wr

# ----------------------------------------
# Contexto del torneo
# ----------------------------------------
dummies_tourney = pd.get_dummies(df['tourney_level'], prefix='tourney')
df = pd.concat([df.reset_index(drop=True), dummies_tourney.reset_index(drop=True)], axis=1)
tourney_columns = dummies_tourney.columns.tolist()

# ----------------------------------------
# Transformar a filas para el modelo
# ----------------------------------------
def crear_fila(p1, p2, row, target):
    base = {
        'p1_name': row[f'{p1}_name'], 'p2_name': row[f'{p2}_name'],
        'p1_rank': row[f'{p1}_rank'], 'p2_rank': row[f'{p2}_rank'],
        'p1_age': row[f'{p1}_age'], 'p2_age': row[f'{p2}_age'],
        'p1_hand': row[f'{p1}_hand'], 'p2_hand': row[f'{p2}_hand'],
        'p1_ace': row[f'{"w" if p1 == "winner" else "l"}_ace'],
        'p2_ace': row[f'{"w" if p2 == "winner" else "l"}_ace'],
        'elo_p1': row[f'elo_{p1}'], 'elo_p2': row[f'elo_{p2}'],
        'h2h_p1_vs_p2': row[f'h2h_{p1}_vs_{p2}'],
        'h2h_p2_vs_p1': row[f'h2h_{p2}_vs_{p1}'],
        'p1_recent_wins': row[f'recent_{p1}_wins'],
        'p2_recent_wins': row[f'recent_{p2}_wins'],
        'p1_surface_wr': row[f'surface_{p1}_wr'],
        'p2_surface_wr': row[f'surface_{p2}_wr'],
        'target': target
    }
    for col in tourney_columns:
        base[col] = row.get(col, 0)
    return base

filas = []
for _, row in df.iterrows():
    filas.append(crear_fila('winner', 'loser', row, 1))
    filas.append(crear_fila('loser', 'winner', row, 0))

df_modelo = pd.DataFrame(filas)
df_modelo = df_modelo[df_modelo['p1_name'].notnull() & df_modelo['p2_name'].notnull()]
df_modelo = df_modelo.reset_index(drop=True)

# ----------------------------------------
# Guardar CSV
# ----------------------------------------
df_modelo.to_csv(os.path.join(folder_path, "tennis_model_dataset.csv"), index=False)
print("✅ Archivo guardado en:", os.path.join(folder_path, "tennis_model_dataset.csv"))
