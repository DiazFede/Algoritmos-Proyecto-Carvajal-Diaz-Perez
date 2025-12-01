# Planificación Multiobjetivo de Observación Satelital

Este repositorio implementa el proyecto **“Planificación Multiobjetivo de Observación Satelital para Cobertura de Eventos Críticos”**. El código modela constelaciones LEO de 5 satélites, genera 60 objetivos críticos con 800 oportunidades de observación y ejecuta las tres etapas descritas en el informe: modelado formal, desarrollo de metaheurísticas evolutivas/híbridas y experimentación comparativa.

## Qué hace este proyecto
- Modela observación satelital como un problema multiobjetivo con tres metas: **cobertura** (se minimiza `-coverage`), **energía** y **makespan** (tiempo hasta cubrir objetivos críticos).
- Genera instancias sintéticas simples: satélites en 2D que se mueven en línea recta con velocidad constante, objetivos fijos, ventanas analíticas `||(p0 - g) + v·t|| <= FOV`, energía proporcional a la duración y cobertura uniforme (`cov = 1`).
- Compara dos algoritmos: baseline **Greedy** y metaheurística **NSGA-II** (pymoo), reparando soluciones para respetar solapes y presupuestos por satélite.
- Produce CSVs, métricas y un dashboard HTML para visualizar frentes y promedios.

## Dependencias y setup rápido
Script recomendado (PowerShell):
```
scripts\setup.ps1
```
Hace:
1) Crea entorno `.venv`  
2) Activa el entorno  
3) Instala `requirements.txt`

Manual (si prefieres):
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo ejecutar experimentos
Ejemplo rápido (pocas semillas para no demorar):
```
python -m src.run_experiments --sat 5 --targets 60 --ops 800 --seeds 5
```
Luego genera el dashboard:
```
python -m src.generate_dashboard
```
Revisa `outputs/`:
- `baseline.csv` (Greedy), `front_nsga2.csv` (frente NSGA-II)
- `summary_metrics.csv`, `success_metrics.json`
- `pareto_greedy_nsga.png` y `dashboard.html`

Parámetros útiles:
- `--pop_size`, `--n_gen`: controlan el costo/beneficio de NSGA-II (por defecto 200/300).
- `--sat`, `--targets`, `--ops`, `--seeds`: tamaño de la instancia y repetición.

## Modelo simplificado (para el informe)
- Satélite: posición inicial `p0`, velocidad constante `v` en 2D durante el horizonte `H=600`.
- Objetivo fijo `(x,y)`. Hay ventana si `||(p0 - g) + v·t|| <= FOV`; se obtiene `t_inicio`, `t_fin` resolviendo una cuadrática.
- Duración = `t_fin - t_inicio`, Energía = `e0 + k·duración`, Cobertura = 1.
- Restricciones aplicadas vía reparación: sin solapes por satélite, sin exceder energía/tiempo por satélite, cada objetivo crítico observado al menos una vez (penalización al makespan si falla).

## Interpretación típica de resultados
- Greedy es rápido y razonable cuando el problema está holgado.
- NSGA-II brinda más cobertura y mejor makespan, a costa de más energía según el plan representativo elegido (por defecto: máxima cobertura con tope 10% sobre energía de Greedy).
- El frente NSGA-II ofrece alternativas; el resumen muestra un solo compromiso.

## Referencias
- Deb, K. (2002). NSGA-II: A fast and elitist multiobjective genetic algorithm. IEEE TEC.
- Glover, F. (1989). Tabu Search. ORSA Journal on Computing.
- Chien, S. et al. (2016). Autonomous planning for spacecraft and satellites. NASA Ames.
- Basso, S. & Pulido, J. (2021). Multiobjective optimization in Earth observation planning. Acta Astronautica.
