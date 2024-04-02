"""

@author: Tarmo Ilves

"""
import numpy as np
import PIL.Image as img

def svd_kuvanpakkaus(kuvatiedosto, k):
    # Kuvan avaaminen
    kuva = img.open(kuvatiedosto)
    # Muunnos numpy-taulukoksi
    kuva_taulukkona = np.array(kuva)
    # Jaetaan värikanavat omiksi taulukoiksi (R, G, B)
    R = kuva_taulukkona[:, :, 0]
    G = kuva_taulukkona[:, :, 1]
    B = kuva_taulukkona[:, :, 2]
    # Käytetään singulaariarvohajotelmaa (SVD) kullekin värikanavalle erikseen
    U_R, Sigma_R, VT_R = np.linalg.svd(R, full_matrices=False)
    U_G, Sigma_G, VT_G = np.linalg.svd(G, full_matrices=False)
    U_B, Sigma_B, VT_B = np.linalg.svd(B, full_matrices=False)
    # Käytetään k ensimmäistä singulaariarvoa ja lasketaan matriisitulo
    R_pakattu = np.dot(U_R[:, :k], np.dot(np.diag(Sigma_R[:k]), VT_R[:k, :]))
    G_pakattu = np.dot(U_G[:, :k], np.dot(np.diag(Sigma_G[:k]), VT_G[:k, :]))
    B_pakattu = np.dot(U_B[:, :k], np.dot(np.diag(Sigma_B[:k]), VT_B[:k, :]))
    # Yhdistetään värikanavat yhtenäiseksi kuvaksi
    pakattu_kuva = np.stack([R_pakattu, G_pakattu, B_pakattu], axis=-1)
    # Varmistetaan, että kaikkien pikselien arvot kuuluvat välille [0, 255]
    pakattu_kuva = np.clip(pakattu_kuva, 0, 255).astype(np.uint8)
    return pakattu_kuva

# jpg-kuvatiedoston sijainti ja käytettävien singulaariarvojen määrä
kuvatiedosto = "kuva.jpg"
k = 200
pakattu_kuva = svd_kuvanpakkaus(kuvatiedosto, k)
# Muunnetaan saatu taulukko kuvaksi tallennettavaan muotoon, ja tallennetaan
tallennettava_kuva = img.fromarray(pakattu_kuva).save(f"svd_pakkaus_k{k}.jpg")
