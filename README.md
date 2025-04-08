# Quijote Simulations & Delaunay

Este es un trabajo que se está realizando en conjunto a [Luna](https://github.com/lunajimenez) con el proposito de analizar las simulaciones de Quijote (ver [aquí](https://quijote-simulations.readthedocs.io/en/latest/)), en concreto, del clúster de Nueva York.

A manera de resumen, se han descargado simulaciones obtenidas del catalogos de Halos (únicamente FoF por el momento), de los datos obtenidos en cada muestra, se han usado todos aquellos que tienen $z = 3$ (o `snapnum = 000`), por tener menor cantidad de datos y ser más tratables computacionalmente hablando. Luego, a partir de los datos se crean los grafos utilizando la triangulación de Delaunay, en estos grafos se sacan algunas métricas y finalmente se grafican y comparan con grafos calculados con Delaunay pero con nodos aleatorios.

En el archivo [001](./001_crear_grafos_fof.ipynb) se leen los datos descargados de FoF para los grafos utilizando la triangulación de Delaunay.  
En el archivo [002](./002_calculate_metrics_based_on_graphs.ipynb) se calculan las métricas de todos los grafos calculados.  
En el archivo [003](./003_plots.ipynb) se realizan gráficas con los datos.

La idea en un futuro será expandir este flujo de trabajo para aplicarlo con Rockstar y no solo con FoF. Además, en las simulaciones de Quijote se debe encontrar una manera de estandarizar los redshift, puesto que algunos tienen más valores de redshift que otros (esta última tiene menos dificultad de realizar, pero igual es importante de realizar).

## Requerimientos

El manejador de paquetes a utilizar no es pip, aunque sí se puede, aquí se está utilizando [uv](https://docs.astral.sh/uv/), así que es necesario instalarlo antes de.

Una vez instalado, solamente se deben ejecutar los siguientes comandos.

``` bash
$uv sync
$source ./.venv/bin/activate
```

El uso de `uv` es simplemente porque es muchísimo más rápido en comparación a pip, pero en caso de no poder descargarlo, está el archivo [requirements.txt](./requirements.txt). Con respecto a esta última simplemente se deben seguir los siguientes comandos.

Crear entorno virtual (opcional, pero recomendable)

``` bash
$python3 -m venv .venv
$source ./.venv/bin/activate
```

Instalar requerimientos

``` bash
$pip install -r requirements.txt
```

## Datos

Desafortunadamente, en este momento no contamos con Git LFS para poder subir los datos al repositorio de Github, sin embargo, son perfectamente descargables en los siguientes enlaces.

Dentro de [quijote/FoF](./quijote/FoF/) se deben descargar los archivos que se encuentran en el siguiente enlace: [https://drive.google.com/drive/folders/1kVdYs3JcDT0_CrdNbxiJchOqfAYyJx-S?usp=sharing](https://drive.google.com/drive/folders/1kVdYs3JcDT0_CrdNbxiJchOqfAYyJx-S?usp=sharing).

Dentro de [quijote/grafos](./quijote/grafos/) se deben descagar los archivos que están en: [https://drive.google.com/drive/folders/1D7uN7TyFe49s_MKvWldYkgKs7powOzRv?usp=sharing](https://drive.google.com/drive/folders/1D7uN7TyFe49s_MKvWldYkgKs7powOzRv?usp=sharing).

Cuando se descargan los datos directamente Drive es posible que se generen varios archivos zip que contienen toda la información, por tanto, el script que se encuentra [aquí](./quijote/unzip_all.sh), puede llegar a ser útil para extraer toda la información dentro de los archivos comprimidos. La manera de usar este script ya está en cada quién, se podría mover el archivo de carpetas o cómo el usuario vea conveniente.

Además, no pasar por alto que se deben remover los archivos `.gitkeep`, porque son únicamente para mostrar carpetas vacías en Github.

## Multicore

Existe un archivo llamado [calc_metrics_multicore.py](./calc_metrics_multicore.py) en el que calcula las métricas de los grafos utilizando paralelismo. A continuación, dejaré un comando que puede llegar a ese útil para utilizarlo:

``` bash
python3 calc_metrics_multicore.py --simus DC_p,DC_m,randoms --snapnum 000 --workers 4 --output ./quijote/grafos/metrics_000.json > output_000_1.log 2>&1 &
```

Los valores de cada uno de los parámetros son de ejemplos, pero la idea principal es la de utilizar un archivo extra para el historial de la salida en consola. Con `>` se estaría sobreescribiendo todo el rato el archivo `output_000_1.log`, redirigiendo al mismo lugar el "Standard Error" y el "Standard Output". Como mencioné antes, no es obligatorio, es solo una opción que escribo: se puede combinar con `tee` para ver la salida en consola también si así se quiere, o evitar que el script no se ejecute en segundo plano quitando el signo `&` al final. Son sugerencias

De igual forma, aquí muestro la información de cada parámetro:

``` bash
$python3 calc_metrics_multicore.py --help

usage: calc_metrics_multicore.py [-h] [--simus SIMUS] [--snapnum {000,001,002,003,004}] [--workers WORKERS] [--output OUTPUT]

Calcular métricas de grafos Delaunay

options:
  -h, --help            show this help message and exit
  --simus SIMUS         Simulación(es) a procesar. Si no se especifica, se procesan todas. Para escoger más de una simulación se debe separar
                        por comas: "simu1,simu2" (default: ""). Simulaciones disponibles: Mnu_pp DC_p Mnu_p DC_m w_m s8_p s8_m w_p
                        fiducial_LR Mnu_ppp randoms fiducial h_m fiducial_HR fiducial_ZA
  --snapnum {000,001,002,003,004}
                        Número de snapshot a procesar (default: 000)
  --workers WORKERS     Número de procesos paralelos (default: 4)
  --output OUTPUT       Ruta para guardar el archivo JSON de métricas (default: ./quijote/grafos/metrics.json)
```

El script automáticamente lista las simulaciones que estén descargadas en el equipo.

<!-- Si alguien llegara a leer esto, la verdad es que no hemos podido lograr que funcione correctamente, hemos utilizado máquinas virtuales de Azure y Google Colab, pero aún no damos, ojalá ya quede cada vez menos! -->

### Comentarios extras

He estado pensando demasiado en el tema de cómo optimizar este código utilizando la paralelización, y no solo con CPU, también con GPU utilizando los CUDA cores, pero no parece haber implementación directa de esto último con la librería iGraph, entonces... ¿qué más se puede hacer?

Porque realmente la implememtación que se tiene puede estar bien, pero no es la mejor, solo se están calculando varios grafos al mismo tiempo, ¿qué tal paralelizar en la manera en que se está calculando el grafo utilizando subgrafos? Esto parece ser una muy buena opción, pero la implementación no la veo tan clara en realidad, porque primero es necesario discutir dónde están los cuellos de botella para ahí calcularlos y luego discutir cómo se haría, aunque parece ser otro tema. Tampoco tiene mucho sentido paralelizar todo, porque con networkx paralelizado para calcular betweenness, sigue siendo relativamente lento a comparación a igraph normal, ¿qué pasaría si paralelizo con igraph? Hice una implementación por mi parte, y no pareció dar mejores resultados, así que no vale la pena intentarlo en todo.

Ahora, si el problema también es de RAM, ¿qué pasa con Dask y su memoria virtual? Es una idea pendiente pero que puede resultar genial, así que dejaré estos comentarios extras para responderme en un fúturo. Y, por lo antes mencionado de CUDA, ¿qué pasa con cugraph? No se ha podido ejecutar bien desde mi computador, pero es una idea que también está pendiente. ¿Se podrá hacer el programa para que utilice cugraph y Dask al mismo tiempo? ¿Será cugraph más lento que igraph? En cualquier caso, igraph es lo mejor en cuanto a performance se refiere, así que mi absoluto respeto para esta módulo.
