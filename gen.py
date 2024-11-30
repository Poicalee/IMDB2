import pandas as pd
import numpy as np

def generuj_dane(n_samples=500):
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

    for col in df.columns[:-1]:
        mask = np.random.rand(len(df)) < 0.2
        df.loc[mask, col] = np.nan

    return df

# Generowanie danych
df = generuj_dane()

# Zapis danych do pliku CSV
df.to_csv('dane_subskrypcji.csv', index=False)
print("Dane zapisane do pliku 'dane_subskrypcji.csv'.")
