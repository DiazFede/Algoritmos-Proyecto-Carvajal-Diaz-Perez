import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
TARGET_HTML = OUT_DIR / "dashboard.html"


def read_csv(name: str) -> pd.DataFrame:
    """
    Lee un CSV de la carpeta de outputs si existe.

    Args:
        name (str): Nombre del archivo CSV.

    Returns:
        pd.DataFrame: Datos leídos o DataFrame vacío si no existe.
    """
    path = OUT_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_success_metrics():
    """
    Carga el JSON de métricas de éxito si existe.

    Returns:
        dict: Métricas cargadas o diccionario vacío.
    """
    path = OUT_DIR / "success_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def scatter_payload(df: pd.DataFrame) -> list:
    """
    Convierte un DataFrame en payload de puntos (x=cobertura, y=energía).

    Args:
        df (pd.DataFrame): DataFrame con coverage_pct y energy.

    Returns:
        list: Lista de dicts con x,y para Chart.js.
    """
    rows = []
    if df.empty:
        return rows
    for _, row in df.iterrows():
        rows.append({"x": float(row["coverage_pct"]), "y": float(row["energy"])})
    return rows


def summary_payload(df: pd.DataFrame) -> dict:
    """
    Convierte resumen en payload para barras.

    Args:
        df (pd.DataFrame): DataFrame con columnas algorithm, coverage_pct, energy, makespan.

    Returns:
        dict: Estructura con labels y listas de valores.
    """
    if df.empty:
        return {"labels": [], "coverage": [], "energy": [], "makespan": []}
    labels = df["algorithm"].tolist()
    cov = df["coverage_pct"].astype(float).tolist()
    energy = df["energy"].astype(float).tolist()
    makespan = df["makespan"].astype(float).tolist()
    return {"labels": labels, "coverage": cov, "energy": energy, "makespan": makespan}


def render_html(data: dict):
    """
    Renderiza el HTML del dashboard y lo guarda en outputs/dashboard.html.

    Args:
        data (dict): Diccionario con scatters, summary y success.

    Returns:
        Path: Ruta al archivo HTML generado.
    """
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Dashboard Planificación Satelital</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;600&display=swap" rel="stylesheet">
  <style>
    body {{
      font-family: 'Roboto', sans-serif;
      margin: 0;
      padding: 0;
      background: #0b1220;
      color: #f2f5ff;
    }}
    header {{
      padding: 24px;
      background: linear-gradient(135deg, #18253a, #10243b);
      border-bottom: 1px solid #243450;
    }}
    h1 {{
      margin: 0;
      font-size: 1.8rem;
    }}
    main {{
      padding: 24px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 24px;
    }}
    section {{
      background: #151f33;
      border: 1px solid #243450;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 10px 20px rgba(0,0,0,0.35);
    }}
    section h2 {{
      margin-top: 0;
      font-size: 1.2rem;
    }}
    canvas {{
      max-width: 100%;
    }}
    ul.meta {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    ul.meta li {{
      margin-bottom: 8px;
    }}
    .code {{
      font-family: monospace;
      background: rgba(255,255,255,0.05);
      padding: 2px 4px;
      border-radius: 4px;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <header>
    <h1>Planificación Multiobjetivo — Dashboard interactivo</h1>
    <p>Greedy (baseline por semilla) vs NSGA-II (frente completo) con resultados de <span class="code">run_experiments.py</span>.</p>
  </header>
  <main>
    <section>
      <h2>Frente de Pareto (Cobertura vs Energía)</h2>
      <canvas id="frontChart" height="320"></canvas>
    </section>
    <section>
      <h2>Resumen Promedios</h2>
      <p style="color:#cfd8f5;margin-top:-8px;font-size:0.9rem;">Cobertura (%), Energía y Makespan para Greedy vs NSGA-II.</p>
      <canvas id="summaryChart" height="320"></canvas>
    </section>
    <section>
      <h2>Métricas de éxito</h2>
      <ul class="meta" id="metrics"></ul>
    </section>
    <section>
      <h2>Glosario rápido</h2>
      <ul class="meta">
        <li><strong>Cobertura (%):</strong> Porcentaje de valor esperado cubierto respecto al máximo de la instancia.</li>
        <li><strong>Energía:</strong> Consumo total de las observaciones aceptadas.</li>
        <li><strong>Makespan:</strong> Tiempo (s) hasta completar la primera observación de cada objetivo crítico.</li>
      </ul>
    </section>
  </main>
  <script>
    const scatterData = {json.dumps(data["scatters"], ensure_ascii=False)};
    const summaryData = {json.dumps(data["summary"], ensure_ascii=False)};
    const successMetrics = {json.dumps(data["success"], ensure_ascii=False, indent=2)};

    const frontCtx = document.getElementById('frontChart').getContext('2d');
    const scatterChart = new Chart(frontCtx, {{
      type: 'scatter',
      data: {{
        datasets: [
          {{
            label: 'NSGA-II (frente Pareto completo)',
            data: scatterData.nsga2,
            backgroundColor: 'rgba(92, 169, 255, 0.8)'
          }},
          {{
            label: 'Greedy (1 plan por semilla)',
            data: scatterData.greedy,
            backgroundColor: 'rgba(124, 230, 135, 1)',
            pointRadius: 7,
            pointStyle: 'triangle'
          }}
        ]
      }},
      options: {{
        scales: {{
          x: {{
            title: {{
              display: true,
              text: 'Cobertura (%)'
            }},
            beginAtZero: false
          }},
          y: {{
            title: {{
              display: true,
              text: 'Energía (unidades)'
            }}
          }}
        }},
        plugins: {{
          legend: {{
            labels: {{
              color: '#f2f5ff'
            }}
          }}
        }}
      }}
    }});

    const summaryCtx = document.getElementById('summaryChart').getContext('2d');
    const summaryChart = new Chart(summaryCtx, {{
      type: 'bar',
      data: {{
        labels: summaryData.labels,
        datasets: [
          {{
            label: 'Cobertura (%)',
            backgroundColor: 'rgba(92, 169, 255, 0.8)',
            data: summaryData.coverage
          }},
          {{
            label: 'Energía (unid.)',
            backgroundColor: 'rgba(255, 199, 101, 0.85)',
            data: summaryData.energy
          }},
          {{
            label: 'Makespan (s)',
            backgroundColor: 'rgba(124, 230, 135, 0.85)',
            data: summaryData.makespan
          }}
        ]
      }},
      options: {{
        responsive: true,
        scales: {{
          x: {{
            ticks: {{ color: '#f2f5ff' }}
          }},
          y: {{
            ticks: {{ color: '#f2f5ff' }}
          }}
        }},
        plugins: {{
          legend: {{
            labels: {{
              color: '#f2f5ff'
            }}
          }},
          tooltip: {{
            callbacks: {{
              label: function(ctx) {{
                return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2);
              }}
            }}
          }}
        }}
      }}
    }});

    const metricsUl = document.getElementById('metrics');
    const labelMap = {{
      nsga_mean_hv: 'HV promedio NSGA-II',
      nsga_global_hv: 'HV global NSGA-II',
      coverage_gain_vs_greedy: 'Ganancia de cobertura vs Greedy (puntos %)',
      energy_delta_vs_greedy: 'Delta de energía vs Greedy (unid.)'
    }};
    const fmt = (v) => {{
      if (v === null || v === undefined || Number.isNaN(v)) return 'N/A';
      if (typeof v === 'number') return v.toFixed(3);
      return v;
    }};
    const entries = Object.entries(successMetrics);
    if (entries.length === 0) {{
      metricsUl.innerHTML = '<li>No se encontraron success_metrics.json</li>';
    }} else {{
      entries.forEach(([key, value]) => {{
        const label = labelMap[key] || key;
        metricsUl.innerHTML += `<li><strong>${{label}}:</strong> ${{fmt(value)}}</li>`;
      }});
    }}
  </script>
</body>
</html>
"""
    TARGET_HTML.write_text(html, encoding="utf-8")
    return TARGET_HTML


def main():
    """Genera el dashboard HTML a partir de los CSV/JSON en outputs."""
    nsga = scatter_payload(read_csv("front_nsga2.csv"))
    greedy = scatter_payload(read_csv("baseline.csv"))
    summary = summary_payload(read_csv("summary_metrics.csv"))
    success = load_success_metrics()

    data = {
        "scatters": {
            "nsga2": nsga,
            "greedy": greedy,
        },
        "summary": summary,
        "success": success,
    }
    path = render_html(data)
    print(f"Dashboard generado en {path}")


if __name__ == "__main__":
    main()
