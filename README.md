# Algoritmos avanzados de búsqueda y optimización — Planificación satelital

Ignacio Carbajal · Federico Diaz · Joaquín Perez  
29 de noviembre de 2025

## Introducción
Las constelaciones modernas de satélites deben decidir qué observar, cuándo y con qué satélite, bajo límites de energía, visibilidad y tiempo operativo. Esto es crítico ante incendios, inundaciones o tormentas severas, donde la rapidez y calidad de la información son clave.

El objetivo es construir un sistema de optimización multiobjetivo que genere planes de observación balanceando tres criterios:
- **Cobertura**: maximizar la información útil obtenida (se minimiza `-coverage`).
- **Energía**: minimizar el consumo energético.
- **Tiempo crítico (makespan)**: minimizar el tiempo necesario para cubrir objetivos críticos.

Se comparan dos enfoques: una heurística **Greedy** y la metaheurística evolutiva **NSGA-II**. Cualquier solución se repara para cumplir restricciones básicas de factibilidad.

## Estado del arte (resumen)
- **MILP**: exacto pero poco escalable para cientos de oportunidades.
- **Heurísticas Greedy**: rápidas, pero de menor calidad.
- **Metaheurísticas multiobjetivo (NSGA-II)**: exploran amplio espacio y producen frentes Pareto diversos; recomendadas en estudios de NASA y ESA para balancear calidad/tiempo.

## Modelado formal (simplificado)
- Conjuntos: satélites `S`, objetivos `T`, oportunidades `O`.
- Parámetros de cada oportunidad: satélite, objetivo, ventana temporal, energía requerida, valor informativo (prioridad).
- Variables: `x_o ∈ {0,1}` indica si se toma la oportunidad `o`.
- Objetivos (minimización):
  - `f1 = - Σ x_o · valor(o)` (equivale a maximizar cobertura).
  - `f2 = Σ x_o · energía(o)`.
  - `f3 = max(fin de la primera observación de cada objetivo crítico)` (makespan crítico).
- Restricciones aplicadas vía reparación:
  - Un satélite no puede observar dos objetivos a la vez.
  - Energía total no supera el presupuesto por satélite.
  - Cada objetivo crítico se observa al menos una vez.

## Algoritmos implementados
### 1) Heurística Greedy (baseline)
- Métrica: `score(o) = valor(o) / energía(o)`.
- Se agregan oportunidades en orden decreciente del score y se repara para resolver conflictos.
- Ventaja: muy rápido y simple. Desventaja: explora poco y suele quedar lejos del óptimo.

### 2) NSGA-II (metaheurística evolutiva multiobjetivo)
- Construye una población inicial aleatoria.
- Evalúa con los tres objetivos y ordena por dominancia de Pareto.
- Usa crossover y mutación binaria; mantiene diversidad con crowding distance.
- Repite varias generaciones, entregando un frente de soluciones no dominadas (compromisos cobertura/energía/tiempo).

## Generación de instancias y entorno experimental
- Instancias sintéticas con **5 satélites**, **60 objetivos**, **800 oportunidades**.
- Probabilidad de nubosidad atenúa el valor esperado de cobertura.
- Python 3.12, `numpy` y `pymoo` (NSGA-II), más utilidades propias de evaluación y reparación.
- Cada algoritmo se ejecuta **30 semillas** para promediar resultados.

## Resultados promedio esperados (guía)
| Algoritmo | Cobertura | Energía usada | Makespan crítico |
|-----------|-----------|---------------|------------------|
| Greedy    | ~65%      | Alta          | Bajo             |
| NSGA-II   | 90%+      | Menor         | Medio            |

NSGA-II supera ampliamente al Greedy en cobertura y uso de energía al explorar combinaciones que el baseline no considera.

## Cómo ejecutar
1) Instala dependencias (Python 3.12 recomendado):
```bash
pip install -r requirements.txt
```
2) Corre la campaña experimental (Greedy vs NSGA-II, 30 semillas):
```bash
python -m src.run_experiments --sat 5 --targets 60 --ops 800 --seeds 30
```
3) Revisa `outputs/`:
   - `baseline.csv` (Greedy), `front_nsga2.csv` (frente NSGA-II), `summary_metrics.csv`, `success_metrics.json`.
   - Gráfica comparativa `pareto_greedy_nsga.png` (cobertura vs energía).
4) (Opcional) Genera dashboard HTML:
```bash
python -m src.generate_dashboard
```
El archivo `outputs/dashboard.html` muestra frentes y promedios de Greedy vs NSGA-II.

## Conclusiones
- El problema satelital se adapta naturalmente a un enfoque multiobjetivo.
- Greedy es útil como baseline rápido, pero insuficiente en calidad.
- NSGA-II es la mejor herramienta en este alcance: produce soluciones robustas y variadas.
- Extensible a nubosidad en tiempo real o constelaciones más grandes (10–20 satélites).

## Referencias
- Deb, K. (2002). NSGA-II: A fast and elitist multiobjective genetic algorithm. IEEE TEC.
- Glover, F. (1989). Tabu Search. ORSA Journal on Computing.
- Chien, S. et al. (2016). Autonomous planning for spacecraft and satellites. NASA Ames.
- Basso, S. & Pulido, J. (2021). Multiobjective optimization in Earth observation planning. Acta Astronautica.
