# assignment-1
meno: Jakub Povinec

použité metódy:
* filtračné metódy (GaussianBlur)
* thresholding, adaptívny thresholding
* detekcia hrán (Canny)
* morfologické operácie (diletation, erosion, closing, opening)
* kontúrová analýza (findContours)
- - - -
## Bubliny
Pri tomto obrázku bolo mojim hlavným cieľom označiť jednotlivé bubliny a zistiť, analyzovať ich plochu. Kód k tomuto obrázku sa nachádza v `bubbles.ipynb`.

### Pokus 1 - Canny
Na začiatku som si načítal daný obrázok a prekonvertoval ho do grayscale.

```
img = cv2.imread('data/pena.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

grayscale bubbles

#### Filtrovanie
Ďalej som sa pomocou `GaussianBlur` snažil vyfiltrovať obrysy bublín, ktoré sa na obrázku nachádzajú vo vnútri povrchových bublín.

```
filtered = cv2.GaussianBlur(gray, (5, 5), 1)
```

V tomto prípade som experimentoval s parametrami danej metódy, kedy sa pri takto zadefinovanom filtri najlepšie identifikujú hrany jednotlivých bublín a taktiež sa odstránia bubliny nachádzajúce sa vo vnútri povrchových bublín.

#### Binarizácia
Na binarizáciu vstupného obrázka som použil **Canny edge detector**.

```
edges = cv2.Canny(filtered, 20, 120)
```

![](README/canny.png)output from canny

Výstup z tohto algoritmu som ďalej upravil pomocou morfologických operácií. Ako prvú som použil **diletáciu**, pomocou ktorej som zvýraznil hrany jednotlivých bublín.

```
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(11, 11))
edges = cv2.dilate(edges, kernel, iterations=1)
```

bubbles diletation

Po diletácií zostali v hranách medzi bublinami 'diery'. Tieto diery, ale nepredstavujú ďalšie bubliny a tak som sa ich rozhodol odstrániť pomocou operácie **closing**.

```
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(11, 11))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
```

bubbles closing

#### Kontúrová analýza
Pred samotným hľadaním obrysov vo vytvorenej maske je potrebné masku invertovať pomocou `cv2.bitwise_not(edges)`.

bubbles inverted

Bez invertovania nefungovala dobre metóda na hľadanie kontúr a napríklad označovala okraj obrázka ako jednu veľkú kontúru, ako je vidieť aj na obrázku nižšie.

bubbles one big contour

Na nájdenie kontúr som použil metódu `findContours` s módom `RETR_EXTERNAL`, ktorý neoznačoval vnútorné bubliny. Následne som len prešiel cez nájdené kontúry, ktoré som zakreslil do pôvodného obrázka a taktiež som si uchoval ich plochu. Na koniec som označil najväčšiu kontúru a vypísal: počet bublín, priemernu plochu a plochu najväčšej bubliny.

```
contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n = 0

areas = []
biggest = { 'area': -1 }
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    areas.append(area)

    if area > biggest['area']:
        biggest['area'] = area
        biggest['contour'] = i

    colour = (randrange(255), randrange(255), randrange(255))
    cv2.drawContours(segmented_img, contours, i, colour, 3)
    n += 1

cv2.drawContours(segmented_img, contours, biggest['contour'], (255, 0, 0), 10)

print('Number of bubbles: ', n)
print(f'Bubbles mean area: {np.mean(areas):.3f}')
print('Bubbles max area: ', biggest['area'])
```

#### Výsledný výstup
bubbles contours canny

```
Canny
-----------------------
Number of bubbles:  735
Bubbles mean area: 12235.473
Bubbles max area:  155223.0
```

### Pokus 2 - Thresholding
Pri bublinách som sa taktiež rozhodol vyskúšať binarizovať vstupný obrázok iným spôsobom, pomocou thresholding-u.

#### Binarizácia
```
edges = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 5, 2)
```

bubbles thresh

V tomto prípade je potrebné použiť adaptívny thresholding, keďže pri použiťí globálneho thresholdingu ovplivňuje výstup aj napr. rozloženie svetla pri fotení.

bubbles global thresh

Ďalej som získanú masku upravil morfologickými operáciami **erode** a **opening**.

```
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(11, 11))
edges = cv2.erode(edges, kernel, iterations=1)

kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(7, 7))
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=2)
```

bubbles thresh mask

#### Kontúrová analýza
Metóda na hľadanie kontúr je rovnaká ako v predchádzajúcom prípade.

#### Výsledný výstup
bubbles contours thresh

```
Thresholding
-----------------------
Number of bubbles:  636
Bubbles mean area: 14054.748
Bubbles max area:  154924.5
```

### Zhodnotenie
Obe metódy na analyzovanie bublín dosiahli porovnateľné výsledky. Obidve označili rovnakú bublinu ako najväčšiu. Na druhú stranu sa líšia v počte označených bublín (rozdiel 99) a rovnako aj v priemernej ploche bublín (rozdiel 1819.275) a v ploche najväčšej bubliny (rozdiel 298.5).

Ako úspešnejšiu považujem prvú metódu, pri ktorej som dokázal presnejšie označiť väčší počet bublín.

## Bunky
V tomto prípade som sa pokúsil vysegmentovať jednotlivé bunky z obrázku `TCGA-18-5592-01Z-00-DX1.tif`. Kód sa nachádza v `cells.ipynb`.

cells

### Pokus 1
Pri tomto pokuse som postupoval podobne ako pri obrázku s bublinami. Najskôr som si prekonvertoval obrázok do grayscale a následne som sa ho pokúsil vyfiltrovať pomocou `cv2.GaussianBlur(gray, (3, 3), 0)`.

#### Binarizácia
V tomto prípade stačilo použiť globálny thresholding.

```
_, mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_OTSU)
```

cells thresh

Na vytvorenej maske som si všimol, že obsahuje pomerne veľa čiernych bodov, ktoré ale nepredstavujú samotné bunky. Na ich odstránenie som použil operáciu **closing** s veľkosťou jadra `(7, 7)`, vďaka ktorej sa mi podarilo obrázok čiastočne vyčistiť.

```
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
```

cells closing

Ako ďalšie som sa rozhodol aplikovať operáciu **erode** aby som aspoň čiastočne zaplnil biele miesta vo vnútri buniek.

```
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
mask = cv2.erode(mask, kernel, iterations=1)
```

cells erode

#### Kontúrová analýza
Pre nájdenie obrysov buniek som najskôr masku invertoval a rovnako ako pri bublinách pomocou metódy `findContours` identifikoval kontúry buniek.

cells inverted

#### Výsledný výstup
Ako je možné vidieť na obrázku nižšie, v tomto prípade sa mi nepodarilo dostatočne oddeliť jednotlivé bunky a kontúry skôr obkreslujú akési zhluky buniek.

cells contours

```
Take 1
-----------------------
Number of cells:  332
```

### Pokus 2
V tomto prípade som namiesto konvertovania pôvodného obrázku, rozdelil obrázok na jednotlivé kanály, z ktorých vyzeral najlepšie práve **červený** kanál.

```
gray = img[:, :, 2] # using only red channel
filtered = cv2.GaussianBlur(gray, (5, 5), 0)
```
cells one channel

#### Binarizácia
Aj v tomto pokuse som použil globálny thresholding a vzniknutú masku som si hneď invertoval.

```
_, mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_OTSU)
mask = cv2.bitwise_not(mask)
```

cells inverted 2

Na masku som následne aplikoval len jednu morfologickú operáciu a to **opening**, vďaka ktorej som odstránil drobné biele body ako aj čiastočne vyplnil vnútro buniek.

```
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(11, 11))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
```

cells opening 2

#### Kontúrová analýza
Kontúrová analýza bola opäť rovnaká ako v predchádzajúcich pokusoch.

#### Výsledný výstup
V tomto prípade sa mi už podarilo označiť väčšie množstvo buniek. Aj tak sa ale v niektorých prípadoch ešte označili väčšie zhluky buniek.

cells contours 2

```
Take 2
-----------------------
Number of cells:  355
```

### Zhodnotenie
Analyzované metódy boli v podstade veľmi podobné. Obe využívali rovnakú filtráciu, binarizáciu a kontúrovú analýzu, a líšili sa hlavne v použitých morfologických operáciach a v spracovaní pôvodného obrázka. Druhá metóda bola nakoniec značne úspešnejšia a označila o **23 buniek viac**.

Napriek tomu, že sa mi podarilo zlepšiť segmentáciu jednotlivých buniek bolo by zaujímavé použiť úplne inú metódu segmentácie, napríklad **watershed** a **distance transform**, ktoré sú vhodné práve v prípadoch kde sa objekty prekrývajú.

### Pokus 3 - Watershed (nedokončené)


#school/pvid