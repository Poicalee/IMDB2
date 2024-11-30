import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)


def generuj_dane(n_samples=500):
    # Generowanie danych jak poprzednio
    wiek = np.random.randint(18, 65, size=n_samples)
    plec = np.random.choice([0, 1], size=n_samples)
    lokalizacja = np.random.choice([1, 2, 3], size=n_samples)
    rodzaj_uslugi = np.random.choice([1, 2, 3], size=n_samples)
    czas_subskrypcji = np.random.randint(1, 24, size=n_samples)
    czas_na_platformie = np.random.randint(10, 100, size=n_samples)
    liczba_reklamacji = np.random.randint(0, 5, size=n_samples)
    srednia_transakcji = np.random.randint(5, 20, size=n_samples)
    srednie_wydatki = np.random.randint(100, 500, size=n_samples)
    zalegle_platnosci = np.random.randint(0, 4, size=n_samples)
    rabaty = np.random.randint(5, 20, size=n_samples)

    porzucenie = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    df = pd.DataFrame({
        "wiek": wiek,
        "płeć": plec,
        "lokalizacja": lokalizacja,
        "rodzaj_uslugi": rodzaj_uslugi,
        "czas_subskrypcji": czas_subskrypcji,
        "czas_na_platformie": czas_na_platformie,
        "liczba_reklamacji": liczba_reklamacji,
        "srednia_transakcji": srednia_transakcji,
        "srednie_wydatki": srednie_wydatki,
        "zalegle_platnosci": zalegle_platnosci,
        "rabaty": rabaty,
        "porzucenie": porzucenie
    })

    # Wprowadzenie losowych braków danych
    for col in df.columns[:-1]:
        mask = np.random.rand(len(df)) < 0.2
        df.loc[mask, col] = np.nan

    return df


# Generowanie danych
df = generuj_dane()

# Śledzenie zmian decyzji w kolejnych krokach
print("ORYGINALNE DANE:")
print(df['porzucenie'].value_counts(normalize=True))

# 1. OBSŁUGA WARTOŚCI BRAKUJĄCYCH - kNN Imputer
knn_imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(
    knn_imputer.fit_transform(df.drop('porzucenie', axis=1)),
    columns=df.columns[:-1]
)
df_imputed['porzucenie'] = df['porzucenie']
print("\nPO IMPUTACJI kNN:")
print(df_imputed['porzucenie'].value_counts(normalize=True))

# 2. NORMALIZACJA - MinMaxScaler
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    minmax_scaler.fit_transform(df_imputed.drop('porzucenie', axis=1)),
    columns=df_imputed.columns[:-1]
)
df_normalized['porzucenie'] = df_imputed['porzucenie']
print("\nPO NORMALIZACJI MinMax:")
print(df_normalized['porzucenie'].value_counts(normalize=True))

# 3. STANDARYZACJA - Z-score
standard_scaler = StandardScaler()
df_standardized = pd.DataFrame(
    standard_scaler.fit_transform(df_normalized.drop('porzucenie', axis=1)),
    columns=df_normalized.columns[:-1]
)
df_standardized['porzucenie'] = df_normalized['porzucenie']
print("\nPO STANDARYZACJI Z-score:")
print(df_standardized['porzucenie'].value_counts(normalize=True))

# 4. DYSKRETYZACJA
discretizer = KBinsDiscretizer(
    n_bins=4,
    encode='ordinal',
    strategy='uniform'
)
df_discretized = pd.DataFrame(
    discretizer.fit_transform(df_standardized.drop('porzucenie', axis=1)),
    columns=df_standardized.columns[:-1]
)
df_discretized['porzucenie'] = df_standardized['porzucenie']
print("\nPO DYSKRETYZACJI:")
print(df_discretized['porzucenie'].value_counts(normalize=True))

# Przygotowanie danych do modelu
X = df_discretized.drop('porzucenie', axis=1)
y = df_discretized['porzucenie']

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Model decyzyjny - Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Predykcja i ewaluacja
y_pred = rf_model.predict(X_test)

# Szczegółowa analiza zmian decyzji
decyzje = pd.DataFrame({
    'Oryginalna etykieta': y_test,
    'Przewidywana etykieta': y_pred
})

# Identyfikacja zmian decyzji
zmiany_decyzji = decyzje[decyzje['Oryginalna etykieta'] != decyzje['Przewidywana etykieta']]

print("\n--- PODSUMOWANIE ZMIAN DECYZJI ---")
print("Całkowita liczba próbek:", len(decyzje))
print("Liczba zmienionych decyzji:", len(zmiany_decyzji))
print("Procent zmienionych decyzji: {:.2f}%".format(len(zmiany_decyzji) / len(decyzje) * 100))

print("\nPrzykłady zmienionych decyzji:")
print(zmiany_decyzji.head())

# Metryki oceny modelu
print("\nDokładność modelu:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nRaport klasyfikacji:\n", classification_report(y_test, y_pred))