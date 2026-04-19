import numpy as np
from datetime import datetime

# ----------------------
# Haversine matrix
# ----------------------
def haversine_matrix(lat_lon):
    R = 6371.0
    lat_rad = np.radians(lat_lon[:, 0])[:, None]
    lon_rad = np.radians(lat_lon[:, 1])[:, None]
    dlat = lat_rad - lat_rad.T
    dlon = lon_rad - lon_rad.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(lat_rad.T) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d

# ----------------------
# Параметры / входные данные
# ----------------------
population_total = 265_000
large_settlements = {"Armavir": 27713, "Vagharshapat": 46356, "Metsamor": 8453}
population_large_total = sum(large_settlements.values())
rural_population = population_total - population_large_total
TOTAL_SHEEP = 139_056

user_input = input("Учитывать уничтожение всего стада при обнаружении болезни? (y/n): ").strip().lower()
ENABLE_CULL = user_input in ["y", "yes", "да"]

village_names = [
    "Aygevan","Araks","Lenughi","Lukashin","Hatsik","Myasnikyan","Sardarapat","Voskehat",
    "Alashkert","Aknalich","Arevik","Argavand","Getashen","Mrgashat","Nalbandyan","Nor_Artagers",
    "Parakar"
]
village_latlon = np.array([
    [40.167000, 43.967000],
    [40.052800, 44.301700],
    [40.125600, 43.965600],
    [40.185100, 44.003300],
    [40.158500, 43.951500],
    [40.177500, 43.906400],
    [40.136111, 44.013889],
    [40.189000, 43.893000],
    [40.112500, 43.950000],
    [40.126000, 43.901000],
    [40.105000, 43.990000],
    [40.140000, 43.980000],
    [40.110000, 43.930000],
    [40.160000, 44.020000],
    [40.180000, 43.970000],
    [40.150000, 43.980000],
    [40.130000, 44.040000],
])
village_populations = np.array([
    1553,1966,1672,2169,2453,4423,6413,3402,2075,3209,3010,2541,2449,5865,4417,1381,5584
], dtype=int)
num_villages = len(village_names)

sheep_per_person = TOTAL_SHEEP / rural_population
sheep_in_village = np.round(village_populations * sheep_per_person).astype(int)
diff_sheep = TOTAL_SHEEP - sheep_in_village.sum()
if diff_sheep != 0:
    sheep_in_village[0] += diff_sheep

dist_matrix = haversine_matrix(village_latlon)

# ----------------------
# Модельные переменные
# ----------------------
S = sheep_in_village.astype(float)
E = np.zeros(num_villages, dtype=float)
I = np.zeros(num_villages, dtype=float)
D = np.zeros(num_villages, dtype=float)
ENV = np.zeros(num_villages, dtype=float)
cull_scheduled = np.zeros(num_villages, dtype=bool)
cull_day = np.full(num_villages, 10**9, dtype=int)

# ----------------------
# Параметры эпидемии
# ----------------------
months_total = 12 * 10
incubation_mean_months = 48.0
p_E_to_I = 1.0 / incubation_mean_months
mean_sick_months = 3.0
p_I_to_D = 1.0 / mean_sick_months
beta_contact = 0.12
beta_env = 0.02
env_per_I = 0.0005
env_decay = 0.02
mig_distance_km = 6.0
base_migration_fraction = 0.005

neighbor_mask = (dist_matrix > 0) & (dist_matrix <= mig_distance_km)
for v in range(num_villages):
    if not neighbor_mask[v].any():
        nearest = int(np.argmin(dist_matrix[v] + np.eye(num_villages)[v] * 1e9))
        neighbor_mask[v, nearest] = True

mig_probs = np.zeros_like(dist_matrix)
for v in range(num_villages):
    neigh = np.where(neighbor_mask[v])[0]
    if len(neigh) > 0:
        mig_probs[v, neigh] = 1.0 / len(neigh)

delay_cull_months = 1
residual_env_after_cull = 0.2

# ----------------------
# Инициализация patient zero (увеличено число зараженных для феерии)
# ----------------------
rng = np.random.default_rng()
init_village = rng.integers(num_villages)
init_count = max(5, int(0.001 * S[init_village]))  # чуть больше зараженных
E[init_village] += init_count
S[init_village] -= init_count
print(f"Patient zero: village {village_names[init_village]} ({init_village}), init E = {init_count}")

# ----------------------
# Хранилище статистики
# ----------------------
history = {
    "S": np.zeros(months_total + 1),
    "E": np.zeros(months_total + 1),
    "I": np.zeros(months_total + 1),
    "D": np.zeros(months_total + 1),
    "ENV_mean": np.zeros(months_total + 1)
}
history["S"][0] = S.sum()
history["E"][0] = E.sum()
history["I"][0] = I.sum()
history["D"][0] = D.sum()
history["ENV_mean"][0] = ENV.mean()

# ----------------------
# Подготовка файла для дампа
# ----------------------
today = datetime.now()
right_now = str(today.time().replace(microsecond=0)).replace(":", "-")
filename = f"generation-{right_now}-{today.day:02d}-{today.month:02d}-{today.year}.txt"
with open(filename, "w") as f:
    f.write(f"Patient zero: {village_names[init_village]}\n")
    f.write("Month - S - E - I - D\n")
    f.write(f"0 - {int(S.sum())} - {int(E.sum())} - {int(I.sum())} - {int(D.sum())}\n")

# ----------------------
# Основной цикл
# ----------------------
for month in range(1, months_total + 1):
    # E -> I
    nE = E.astype(int)
    trans_EI = rng.binomial(nE, p_E_to_I) if nE.sum() > 0 else np.zeros_like(E, dtype=int)
    E = E - trans_EI
    I = I + trans_EI

    # I -> D
    nI = I.astype(int)
    trans_ID = rng.binomial(nI, p_I_to_D) if nI.sum() > 0 else np.zeros_like(I, dtype=int)
    I = I - trans_ID
    D = D + trans_ID

    # ENV
    ENV = ENV + env_per_I * I
    ENV = np.clip(ENV, 0.0, 1.0)
    ENV = ENV * (1.0 - env_decay)

    # Contact infection
    N_v = (S + E + I)
    density_factor = np.where(N_v > 0, I / N_v, 0.0)
    p_contact = np.clip(beta_contact * density_factor, 0.0, 1.0)
    new_from_contact = rng.binomial(S.astype(int), p_contact)
    S = S - new_from_contact
    E = E + new_from_contact

    # Environment infection
    p_env = np.clip(beta_env * ENV, 0.0, 1.0)
    new_from_env = rng.binomial(S.astype(int), p_env)
    S = S - new_from_env
    E = E + new_from_env

    # Migration
    mig_out_frac = base_migration_fraction
    out_S = rng.binomial(S.astype(int), mig_out_frac)
    out_E = rng.binomial(E.astype(int), mig_out_frac)
    out_I = rng.binomial(I.astype(int), mig_out_frac)
    incoming_S = np.zeros_like(S)
    incoming_E = np.zeros_like(E)
    incoming_I = np.zeros_like(I)
    for v in range(num_villages):
        neigh = np.where(mig_probs[v] > 0)[0]
        if len(neigh) == 0: continue
        probs = mig_probs[v, neigh] / mig_probs[v, neigh].sum()
        if out_S[v] > 0: incoming_S[neigh] += rng.multinomial(int(out_S[v]), probs)
        if out_E[v] > 0: incoming_E[neigh] += rng.multinomial(int(out_E[v]), probs)
        if out_I[v] > 0: incoming_I[neigh] += rng.multinomial(int(out_I[v]), probs)
    S = S - out_S + incoming_S
    E = E - out_E + incoming_E
    I = I - out_I + incoming_I

    # Schedule cull
    if ENABLE_CULL:
        newly_clinical = (I > 0) & (~cull_scheduled)
        for v in np.where(newly_clinical)[0]:
            cull_scheduled[v] = True
            cull_day[v] = month + delay_cull_months
            ENV[v] = min(1.0, ENV[v] + 0.1)

    # Execute culls
    if ENABLE_CULL:
        to_cull = np.where((cull_scheduled) & (month >= cull_day))[0]
        for v in to_cull:
            D[v] += S[v] + E[v] + I[v]
            S[v] = E[v] = I[v] = 0.0
            ENV[v] *= residual_env_after_cull
            cull_scheduled[v] = False
            cull_day[v] = 10**9

    # Save stats
    history["S"][month] = S.sum()
    history["E"][month] = E.sum()
    history["I"][month] = I.sum()
    history["D"][month] = D.sum()
    history["ENV_mean"][month] = ENV.mean()

    # Dump to file
    with open(filename, "a") as f:
        f.write(f"{month} - {int(S.sum())} - {int(E.sum())} - {int(I.sum())} - {int(D.sum())}\n")

# ----------------------
# Итоги
# ----------------------
print("\n--- Результаты симуляции ---")
print(f"Период моделирования: {months_total} месяцев (~{months_total/12:.1f} лет)")
print(f"Всего овец (старт): {TOTAL_SHEEP:,}")
print(f"Проверка суммы: S+E+I+D = {history['S'][-1] + history['E'][-1] + history['I'][-1] + history['D'][-1]:,.0f}\n")
print("Итог через", months_total, "месяцев:")
print(f"  Здоровые (S): {history['S'][-1]:.0f}")
print(f"  В инкубации (E): {history['E'][-1]:.0f}")
print(f"  Клинические (I): {history['I'][-1]:.0f}")
print(f"  Мёртвые/убитые (D): {history['D'][-1]:.0f}")

peak_month = int(np.argmax(history['I']))
print(f"\nПик клинических I: месяц {peak_month} (через {peak_month/12:.1f} лет), I_peak = {history['I'][peak_month]:.0f}")

check_months = [12, 36, 60, months_total]
for m in check_months:
    print(f"Месяц {m}: S={history['S'][m]:.0f}, E={history['E'][m]:.0f}, I={history['I'][m]:.0f}, D={history['D'][m]:.0f}")

print(f"\nСтатистика по месяцам сохранена в файл: {filename}")
