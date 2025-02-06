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

Además, no pasar por alto que se deben remover los archivos `.gitkeep`, porque son únicamente para mostrar carpetas vacías en Github.
