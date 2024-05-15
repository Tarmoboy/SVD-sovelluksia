'''
Tiedoston nimi: svd-elokuvasuosittelu.py
Tekijä: Tarmo Ilves
Viimeksi muokattu: 16.5.2024
Kuvaus: Elokuvasuositusten tekeminen singulaariarvohajotelman avulla. 
        Mahdollisuus myös iteroimalla sekä regularisointikertoimella  
        parantaa saatavia suosituksia. Esimerkkinä toiminnasta käytetty 
        MovieLens Latest Small -tietoaineistoa.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#import matplotlib.pyplot as plt

def alusta_rivikeskiarvoilla(arvostelutaulukko):
    '''
    Muuttaa arvostelutaulukon matriisiksi ja täydentää sen puuttuvat arvot 
    kunkin rivin keskiarvolla. 

    Parametrit
    ----------
    arvostelutaulukko : pandas.DataFrame
        Taulukko, joka sisältää puuttuvia alkioita.

    Palauttaa
    ---------
    R_x : numpy.ndarray
        Arvostelutaulukosta luotu matriisi, jonka puuttuvat alkiot on 
        täydennetty rivikeskiarvoilla.
    puuttuvat : numpy.ndarray
        Boolean-matriisi, joka kertoo puuttuiko kunkin indeksin alkio vai ei
    '''
    # Matriisimuotoon
    R = arvostelutaulukko.values
    # Puuttuvista arvoista boolean-matriisi
    puuttuvat = np.isnan(R)
    # Rivikeskiarvojen laskeminen
    rivikeskiarvot = arvostelutaulukko.mean(axis=1, skipna=True)
    # Rivikeskiarvojen syöttö taulukkoon
    arvostelutkeskiarvoilla = arvostelutaulukko.T.fillna(rivikeskiarvot).T
    # Taulukko matriisimutoon
    R_x = arvostelutkeskiarvoilla.values
    return R_x, puuttuvat

def svd_puuttuvien_iterointi(R, puuttuvat, k=5, iteroi=1, gamma=0):
    '''
    Laskee singulaariarvohajotelman (SVD) matriisille R ja päivittää vain
    siitä puuttuvat arvot iteroimalla.

    Parametrit
    ----------
    R : numpy.ndarray
        Matriisi, jolle halutaan tehdä SVD.
    puuttuvat : numpy.ndarray
        Boolean-matriisi, joka sisältää tiedon alun perin matriisista R 
        puuttuneista alkioista.
    k : int, valinnainen
        Approksimaatiossa käytettävien singulaariarvojen lukumäärä.
        Oletuksena k=5.
    iteroi : int, valinnainen
        Kuinka monta kertaa halutaan laskea puuttuvat arvot uusiksi. 
        Oletuksena lasketaan vain kerran.
    gamma : float, valinnainen
        Regularisointikerroin, joka vähennetään jokaisesta singulaariarvosta
        ennen matriisin uudelleenluomista. Oletuksena kerrointa ei käytetä.

    Palauttaa
    ---------
    approksimaatiot : pandas.DataFrame
        Matriisin R approksimaatio taulukkomuodossa.
    '''
    R_i = R.copy()
    for _ in range(iteroi):
        # Kompakti SVD
        U, Sigma, VT = np.linalg.svd(R_i, full_matrices=False)
        # Regularisaatiokertoimen vähentäminen singulaariarvoista
        Sigma_reg = np.maximum(Sigma - gamma, 0)
        # Approksimaatio astetta k
        R_k = np.dot(U[:, :k], np.dot(np.diag(Sigma_reg[:k]), VT[:k, :]))
        # Vain puuttuvat arvot päivitetään
        R_i[puuttuvat] = R_k[puuttuvat]
    # Muunnos taulukoksi
    approksimaatiot = pd.DataFrame(R_i, index=arvostelutaulukko.index, 
                                   columns=arvostelutaulukko.columns)
    return approksimaatiot

def laske_virheet(approksimaatiot, testijoukko):
    '''
    Laskee RMSE ja MAE testijoukon ja ennustettujen arvostelujen välillä.
    
    Parametrit
    ----------
    approksimaatiot : pandas.DataFrame
        SVD:n avulla lasketut approksimaatiot puuttuneille alkioille.
    testijoukko : pandas.DataFrame
        Testidata, joka sisältää todelliset arvostelut.
    
    Palauttaa
    ---------
    rmse : float
        Laskettu keskineliövirheen neliöjuuri (RMSE).
    mae : float
        Laskettu keskimääräinen absoluuttinen virhe (MAE).
    '''
    # Muutetaan approksimaatiot-taulukosta rating -> prediction
    ennusteet = approksimaatiot.stack().reset_index()
    ennusteet.columns = ['userId', 'movieId', 'prediction']
    # Yhdistetään testijoukko ja ennusteet
    yhteenveto = pd.merge(testijoukko, ennusteet, 
                          on=['userId', 'movieId'], how='inner')
    # Lasketaan virheet
    rmse = np.sqrt(mean_squared_error(yhteenveto['prediction'], 
                                      yhteenveto['rating']))
    mae = mean_absolute_error(yhteenveto['prediction'], 
                              yhteenveto['rating'])
    return rmse, mae

def suosittele(approksimaatiot, userId, elokuvat, arvostelut, suosit=20):
    '''
    Antaa elokuvasuosituksia SVD-approksimaatioon perustuen. 

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
    suosit : int, valinnainen
        Kuinka monta elokuvasuositusta halutaan palautettavan, oletus 20.

    Palauttaa
    ---------
    u_arvostelut : pandas.DataFrame
        Kaikki valitun käyttäjän antamat elokuva-arvostelut.
    suositukset : pandas.DataFrame
        Elokuvasuosituksia valitulle käyttäjälle.

    '''
    # Ennustetut arvostelut käyttäjälle userId
    u_ennusteet = approksimaatiot.loc[userId].sort_values(ascending=False)
    # Käyttäjän jo olemassa olevat arvostelut
    u_arvostelut = arvostelut[arvostelut.userId == userId]
    u_arvostelut = u_arvostelut.merge(elokuvat, how='left', 
                                      left_on='movieId', 
                                      right_on='movieId').sort_values(
                                          ['rating'], ascending=False)   
    # Tarpeeton userId -sarake pois
    u_arvostelut = u_arvostelut[['movieId', 'rating', 'title', 'genres']]
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
        'prediction', ascending=False).head(suosit)
    # Tulostuksen yksinkertaistamiseksi indeksoinnit movieId mukaan
    u_arvostelut = u_arvostelut.set_index('movieId')
    suositukset = suositukset.set_index('movieId')
    return u_arvostelut, suositukset

# Käyttäjien elokuva-arvostelut sisältävän tiedoston luku
arvostelut = pd.read_csv('ml-latest-small/ratings.csv', sep=',', 
                         usecols=['userId', 'movieId', 'rating'])
# Arvosteludatan jakaminen opetus- ja testijoukkoihin 75 % - 25 %
opetusjoukko_i, testijoukko_i = train_test_split(arvostelut.index, 
                                                 test_size=0.25, 
                                                 random_state=42)
# Opetusjoukon taulukon koko vastaamaan arvostelut-taulukkoa
opetusjoukko = arvostelut.copy()
# Testijoukossa sijaitsevat arvostelut määrittelemättömiksi
opetusjoukko.loc[testijoukko_i, 'rating'] = np.nan
# Testijoukon luonti indeksien perusteella
testijoukko = arvostelut.loc[testijoukko_i]
# Elokuvatiedot sisältävän tiedoston luku
elokuvat = pd.read_csv('ml-latest-small/movies.csv', sep=',',
                       usecols=['movieId', 'title', 'genres'])
# Enemmän sarakkeita näkyviin
pd.set_option('display.max_columns', 8)
# Enemmän merkkejä tulostusleveyteen
pd.set_option('display.width', 150)
# Sarakkeille sopiva leveys gradua varten, muuta tarvittaessa
pd.set_option('display.max_colwidth', 30)
# Muunnos taulukoksi, jossa riveinä käyttäjät ja sarakkeina elokuvat
arvostelutaulukko = opetusjoukko.pivot(index='userId', columns='movieId', 
                                       values='rating')
# Muunnos matriisiksi ja puuttuvien alkioiden etsiminen
arvostelumatriisi, puuttuvat = alusta_rivikeskiarvoilla(arvostelutaulukko)
# Singulaariarvojen piirtäminen kuvaajaan
#_, Sigma, _ = np.linalg.svd(arvostelumatriisi, full_matrices=False)
#plt.figure(figsize=(6, 5))
#plt.plot(Sigma, marker='x', linestyle='none', color='deeppink')
#plt.title('Singulaariarvojen kuvaaja')
#plt.xlabel('Indeksi')
#plt.ylabel('Singulaariarvon suuruus')
#plt.yscale('log')
#plt.grid(True, which='major', linewidth=0.5)
#plt.grid(which='minor', linestyle='--', linewidth=0.5)
#plt.show()
# Iteratiivinen SVD
approksimaatiot = svd_puuttuvien_iterointi(arvostelumatriisi, puuttuvat, 
                                           iteroi=5, k=25, gamma=15.6)
# RMSE ja MAE laskeminen
rmse, mae = laske_virheet(approksimaatiot, testijoukko)
# Käyttäjän id:n valinta
kayttaja = 123
# Elokuvasuositusten lukumäärä
suosituksia = 20
# Suosittele-funktion käyttö
arvosteltu, suositukset = suosittele(approksimaatiot, kayttaja, elokuvat, 
                                     arvostelut, suosituksia)
print(f'RMSE: {rmse}')
print(f' MAE: {mae}')
print('------------------------\n')
print(f'Käyttäjän id {kayttaja} parhaiten arvostelemat elokuvat, ' 
      '20 ensimmäistä')
print('----------------------------------------------------------------')
print(arvosteltu.head(20), '\n')
print(f'Käyttäjälle id {kayttaja} suositeltavia elokuvia, {suosituksia} ' 
      'suositelluinta')
print('------------------------------------------------------------')
print(suositukset)
