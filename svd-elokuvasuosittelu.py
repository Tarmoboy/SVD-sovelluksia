"""

@author: Tarmo Ilves

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#import matplotlib.pyplot as plt
#from scipy.sparse.linalg import svds

def alusta_rivikeskiarvoilla(arvostelutaulukko):
    """
    Muuttaa arvostelutaulukon matriisiksi ja täydentää sen puuttuvat arvot 
    kunkin rivin keskiarvolla. 

    Parametrit
    ----------
    arvostelutaulukko : pandas.DataFrame
        Taulukko, joka sisältää puuttuvia alkioita.

    Palauttaa
    -------
    R_x : numpy.ndarray
        Arvostelutaulukosta luotu matriisi, jonka puuttuvat alkiot on 
        täydennetty rivikeskiarvoilla.
    puuttuvat : numpy.ndarray
        Boolean-matriisi, joka kertoo puuttuiko kunkin indeksin alkio vai ei
    """
    # Matriisimuotoon
    R = arvostelutaulukko.values
    # Puuttuvista arvoista boolean-matriisi
    puuttuvat = np.isnan(R)
    rivikeskiarvot = arvostelutaulukko.mean(axis=1, skipna=True)
    arvostelutkeskiarvoilla = arvostelutaulukko.T.fillna(rivikeskiarvot).T
    R_x = arvostelutkeskiarvoilla.values
    return R_x, puuttuvat

def svd_puuttuvien_iterointi(R, puuttuvat, iterointikierroksia=1, k=5):
    """
    Laskee singulaariarvohajotelman (SVD) matriisille R ja päivittää vain
    siitä puuttuvat arvot iteroimalla.

    Parametrit
    ----------
    R : numpy.ndarray
        Matriisi, jolle halutaan tehdä SVD.
    puuttuvat : numpy.ndarray
        Boolean-matriisi, joka sisältää tiedon alun perin matriisista R 
        puuttuneista alkioista.
    iterointikierroksia : int, valinnainen
        Kuinka monta kertaa halutaan laskea puuttuvat arvot uusiksi. 
        Oletuksena lasketaan vain kerran.
    k : int, valinnainen
        Approksimaatiossa käytettävien singulaariarvojen lukumäärä.
        Oletuksena k=5.

    Palauttaa
    ----------
    R_i : numpy.ndarray
        Matriisin R approksimaatio.
    """
    R_i = R.copy()
    for _ in range(iterointikierroksia):
        # Kompakti SVD
        U, Sigma, VT = np.linalg.svd(R_i, full_matrices=False)
        #U, Sigma, VT = svds(R_i, k)    # jos isompi datasetti
        # Approksimaatio astetta k
        R_k = np.dot(U[:, :k], np.dot(np.diag(Sigma[:k]), VT[:k, :]))
        # Vain puuttuvat arvot päivitetään
        R_i[puuttuvat] = R_k[puuttuvat]
    return R_i

def laske_virheet(testijoukko, approksimaatiot):
    """
    Laskee RMSE:n ja MAE:n testijoukon ja ennustettujen arvostelujen välillä.
    
    Parametrit
    ----------
    testijoukko : pandas.DataFrame
        Testidatasetti, joka sisältää todelliset arvostelut.
    approksimaatiot : pandas.DataFrame
        SVD:n avulla lasketut approksimaatiot puuttuneille alkioille.
    
    Palauttaa
    -------
    rmse : float
        Laskettu keskineliövirheen neliöjuuri (RMSE).
    mae : float
        Laskettu keskimääräinen absoluuttinen virhe (MAE).
    """
    # Muutetaan approksimaatiot-taulukosta rating -> prediction
    ennusteet = approksimaatiot.stack().reset_index()
    ennusteet.columns = ['userId', 'movieId', 'prediction']

    # Yhdistetään testijoukko ja ennusteet
    yhteenveto = pd.merge(testijoukko, ennusteet, 
                          on=['userId', 'movieId'], how='inner')

    # Lasketaan virheet
    rmse = np.sqrt(mean_squared_error(yhteenveto['rating'], 
                                      yhteenveto['prediction']))
    mae = mean_absolute_error(yhteenveto['rating'], 
                              yhteenveto['prediction'])
    return rmse, mae

def suosittele(approksimaatiot, userId, elokuvat, arvostelut, suosituksia=20):
    """
    

    Parametrit
    ----------
    approksimaatiot : pandas.DataFrame
        SVD:n antamat approksimaatiot muutettuna taulukoksi. 
    userId : int
        Käyttäjän userId, jolla tietoa etsitään taulukoista.
    elokuvat : pandas.DataFrame
        Tiedon elokuvista sisältävä taulukko, tarvittavat sarakkeet ovat 
        'movieId', 'title', 'genres'.
    arvostelut : pandas.DataFrame
        Alkuperäiset elokuva-arvostelut sisältä taulukko, tarvittavat 
        sarakkeet ovat 'userId', 'movieId', 'rating'.
    suosituksia : int, valinnainen
        Kuinka monta elokuvasuositusta halutaan palautettavan, oletuksena 20.

    Palauttaa
    -------
    u_arvostelut : pandas.DataFrame
        Kaikki valitun käyttäjän antamat elokuva-arvostelut.
    suositukset : pandas.DataFrame
        Elokuvasuosituksia valitulle käyttäjälle.

    """
    # Ennustetut arvostelut käyttäjälle userId
    u_ennusteet = approksimaatiot.loc[userId].sort_values(ascending=False)

    # Käyttäjän jo olemassa olevat arvostelut
    u_arvostelut = arvostelut[arvostelut.userId == userId]
    u_arvostelut = u_arvostelut.merge(elokuvat, how='left', 
                                      left_on='movieId', 
                                      right_on='movieId').sort_values(
                                          ['rating'], ascending=False)
    
    # Ennusteiden yhdistäminen elokuvatietoihin
    suositukset = elokuvat.merge(pd.DataFrame(u_ennusteet).reset_index(), 
                                 how='left', left_on='movieId', 
                                 right_on='movieId').rename(
                                     columns={userId: 'prediction'})
    
    # Jätetään pois käyttäjän jo arvostelemat elokuvat
    suositukset = suositukset[~suositukset['movieId'].isin(
        u_arvostelut['movieId'])]
    
    # Sarakkeiden järjestys vastaamaan arvostelutaulukkoa
    sarakkeet = ['movieId', 'prediction', 'title', 'genres']
    
    # Suositukset suuruusjärjestykseen ja valitaan ensimmäiset
    suositukset = suositukset[sarakkeet].sort_values(
        'prediction', ascending=False).head(suosituksia)
    
    return u_arvostelut, suositukset

# Vanhan 1m-datan lukemiseen
#arvostelut = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python', 
#                         header=None, usecols=[0, 1, 2], 
#                         names=['userId', 'movieId', 'rating'],
#                         encoding='ISO-8859-1')

#elokuvat = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python', 
#                       header=None, usecols=[0, 1, 2], 
#                       names=['movieId', 'title', 'genres'],
#                       encoding='ISO-8859-1')

# Käyttäjien elokuva-arvostelut sisältävän tiedoston luku
arvostelut = pd.read_csv('ml-latest-small/ratings.csv', sep=",", 
                         usecols=['userId', 'movieId', 'rating'])

# Arvosteludatan jakaminen opetus- ja testijoukkoihin 75 % - 25 %
opetusjoukko, testijoukko = train_test_split(arvostelut, test_size=0.25, 
                                             random_state=42)

# Elokuvatiedot sisältävän tiedoston luku
elokuvat = pd.read_csv('ml-latest-small/movies.csv', sep=",",
                       usecols=['movieId', 'title', 'genres'])

# Enemmän sarakkeita näkyviin
pd.set_option('display.max_columns', 8)

# Enemmän merkkejä tulostusleveyteen
pd.set_option('display.width', 150)

#print(elokuvat.head())

#print(arvostelut.info())
#print(arvostelut.head())
#print(opetusjoukko.head())
#print(testijoukko.head())

#n_kayttajat = opetusjoukko.userId.unique().shape[0]
#n_elokuvat = opetusjoukko.movieId.unique().shape[0]

#print(n_kayttajat, n_elokuvat)

# Muunnos taulukoksi, jossa riveinä käyttäjät ja sarakkeina elokuvat
arvostelutaulukko = opetusjoukko.pivot(index='userId', columns='movieId', 
                                       values='rating')

#print(arvostelutaulukko.head())

# Muunnos matriisiksi ja puuttuvien alkioiden etsiminen
arvostelumatriisi, puuttuvat = alusta_rivikeskiarvoilla(arvostelutaulukko)

#plt.figure(figsize=(10, 6))
#plt.plot(sigma, marker='o', linestyle='-', color='b')
#plt.title('Singulaariarvojen Kuvaaja')
#plt.xlabel('Singulaariarvon indeksi')
#plt.ylabel('Singulaariarvon suuruus')
#plt.yscale('log') # Logaritminen asteikko y-akselille
#plt.grid(True, which="both", ls="--", linewidth=0.5)
#plt.show()

# Iteratiivinen SVD
R = svd_puuttuvien_iterointi(arvostelumatriisi, puuttuvat, 
                             iterointikierroksia=5, k=5)
# Pienin rmse ja mae arvolla k=2

# Muunnos takaisin matriisista taulukoksi
approksimaatiot = pd.DataFrame(R, index=arvostelutaulukko.index, 
                               columns=arvostelutaulukko.columns)

# RMSE ja MAE laskeminen
rmse, mae = laske_virheet(testijoukko, approksimaatiot)

# Käyttäjän id:n valinta
kayttaja = 123
# kayttaja = 591
# kayttaja = 248
# kayttaja = 605
# kayttaja = 274

# Elokuvasuositusten lukumäärä
suosituksia = 20

arvosteltu, suositukset = suosittele(approksimaatiot, kayttaja, elokuvat, 
                                   arvostelut, suosituksia)
print(f"RMSE: {rmse}")
print(f" MAE: {mae}")
print("--------------------------------------------------\n")
print(f"Käyttäjän id {kayttaja} parhaiten arvostelemat elokuvat," 
      "20 ensimmäistä")
print("--------------------------------------------------")
print(arvosteltu.head(20))
print("--------------------------------------------------\n")

print(f"Käyttäjälle id {kayttaja} suositeltavia elokuvia, {suosituksia}" 
      " suositelluinta")
print("--------------------------------------------------")
print(suositukset)


# Halutessa vertaus Surprisen Funk SVD -algoritmiin
#from surprise import SVD
#from surprise.model_selection import cross_validate
#from surprise import Dataset, Reader
#reader = Reader(rating_scale=(1, 5))

# Funk SVD
#algo = SVD()
#data = Dataset.load_from_df(arvostelut[["userId", "movieId", "rating"]], reader)
#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=4, verbose=True)







                                                