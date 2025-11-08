# Planificación Multiobjetivo de Observación Satelital

Este repositorio implementa el proyecto **“Planificación Multiobjetivo de Observación Satelital para Cobertura de Eventos Críticos”**. El código modela constelaciones LEO de 5 satélites, genera 60 objetivos críticos con 800 oportunidades de observación y ejecuta las tres etapas descritas en el informe: modelado formal, desarrollo de metaheurísticas evolutivas/híbridas y experimentación comparativa.

## Alcance del proyecto
- **Modelado formal** con restricciones de energía, ventanas temporales, maniobras (Δθ) y cobertura mínima de objetivos críticos (m_t ≥ 1).
- **Función multiobjetivo**: maximizar cobertura priorizada (se minimiza `-coverage`), minimizar consumo energético y minimizar el makespan sobre zonas críticas.
- **Metaheurísticas**: heurística greedy (baseline), NSGA-II, MOSA y refinamiento híbrido NSGA-II + Tabu Search.
- **Simulación y resultados**: generador de instancias, ejecución multi-seed (30 corridas) y cálculo de métricas (cobertura %, energía, tiempo, hipervolumen normalizado).

## Ejecución
1. Instala dependencias (Python 3.12 recomendado):
   ```
   pip install -r requirements.txt
   ```
2. Lanza la campaña experimental alineada con el informe (5 satélites, 60 objetivos, 800 oportunidades, 30 semillas):
   ```
   python -m src.run_experiments --sat 5 --targets 60 --ops 800 --seeds 30
   ```
3. Revisa `outputs/` para encontrar:
   - `baseline.csv`, `front_nsga2.csv`, `mosa.csv`, `tabu.csv`
   - `summary_metrics.csv` (promedios de cobertura, energía, tiempo y HV)
   - `success_metrics.json` (≥90 % cobertura crítica, reducción de energía vs. greedy, hipervolumen global)
   - Gráficas: `pareto_nsga2.png`, `pareto_mosa.png`, `pareto_all.png`
4. (Opcional) Genera un dashboard HTML interactivo para presentar los resultados:
   ```
   python -m src.generate_dashboard
   ```
   El archivo `outputs/dashboard.html` incluye los frentes de Pareto, los promedios por algoritmo y las métricas de éxito listas para incrustar en el informe.

## Resultados reportados en el documento
| Algoritmo         | Cobertura (%) | Energía (unid.) | Tiempo (s) | Hipervolumen |
|-------------------|---------------|-----------------|------------|--------------|
| Greedy            | 65.2          | 100             | 2.3        | 0.42         |
| MOSA              | 78.5          | 83              | 15.6       | 0.67         |
| NSGA-II           | 89.7          | 76              | 24.1       | 0.82         |
| NSGA-II + Tabú    | 91.4          | 73              | 28.5       | 0.86         |

Estos valores sirven como referencia: la tubería `run_experiments` replica la configuración (30 semillas) y registra las métricas que alimentan la tabla anterior. El umbral de éxito descrito en el informe (≥90 % cobertura, ≥20 % reducción energética frente a greedy y hipervolumen > 0.75) se verifica automáticamente en `success_metrics.json`.

## Componentes principales
- `src/data.py`: generador de instancias (5 satélites, 60 objetivos, 800 oportunidades) con prioridades, nubosidad y presupuestos energéticos/temporales.
- `src/evaluate.py`: calcula los tres objetivos, repara solapes, penaliza observaciones críticas no atendidas y expone métricas auxiliares.
- `src/greedy.py`: heurística baseline valor/energía y wrapper `run_baseline` reutilizable entre semillas.
- `src/nsga2_run.py`: NSGA-II (población 200, 300 generaciones, crossover 0.8, mutación 0.1) con capacidad de inyectar instancias compartidas.
- `src/mosa.py`: implementación de MOSA con vecindarios flip/swap y enfriamiento progresivo.
- `src/tabu.py`: mejora local enfocada en reducir energía manteniendo cobertura (cov_tolerance configurable).
- `src/run_experiments.py`: orquesta las 30 corridas, calcula hipervolumen normalizado, selecciona soluciones representativas y genera reportes/figuras.

## Notas y próximos pasos
- Todos los algoritmos minimizan (`neg_coverage`, energía, makespan). La cobertura priorizada se interpreta como `%` respecto al máximo esperado de la instancia.
- Las gráficas usan cobertura (%) vs. energía para facilitar la comparación visual mencionada en la discusión del informe.
- Las líneas futuras propuestas (aprendizaje para ajustar órbitas, nubosidad en tiempo real, constelaciones de 20 satélites) pueden montarse sobre el generador y las mismas plantillas de evaluación.
