// ✅ Register matrix chart plugin if available
function matrixPluginLoaded() {
  const registry = Chart.registry && Chart.registry.controllers;
  return (
    registry &&
    (registry.has?.("matrix") ||
      (typeof registry.get === "function" && registry.get("matrix"))) ||
    false
  );
}

if (!matrixPluginLoaded()) {
  if (
    window.chartjsChartMatrix &&
    window.chartjsChartMatrix.MatrixController &&
    window.chartjsChartMatrix.MatrixElement
  ) {
    try {
      Chart.register(
        window.chartjsChartMatrix.MatrixController,
        window.chartjsChartMatrix.MatrixElement
      );
    } catch (err) {
      console.warn("Matrix plugin registration failed", err);
    }
  }

  if (!matrixPluginLoaded()) {
    console.warn(
      "chartjs-chart-matrix plugin not found; using canvas fallback for heatmap"
    );
  }
}

// 🧠 Listen for routing form submit
document.getElementById("routingForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const expertIds = document.getElementById("experts").value.split(",").map(Number);
  const tokenIds = document.getElementById("tokens").value.split(",").map(Number);
  const useQuantum = document.getElementById("useQuantum").checked;

  const model_graph = { experts: expertIds };
  const token_stream = tokenIds.map(id => ({ token_id: id }));

  fetch("/q-routing", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_graph, token_stream, use_quantum: useQuantum })
  })
    .then(res => {
      if (!res.ok) throw new Error("Routing failed.");
      return res.json();
    })
    .then(data => {
      const outputDiv = document.getElementById("output");
      outputDiv.innerHTML = `
        <h3>Routing Results</h3>
        <p><strong>Status:</strong> ${data.status}</p>
        <p><strong>Mode:</strong> ${data.mode}</p>
        <p><strong>Routing Score:</strong> ${data.routing_score}</p>
        <pre><strong>Params:</strong> ${JSON.stringify(data.optimized_params, null, 2)}</pre>
      `;

      const assignments = data.results.assignments || data.results;
      if (!Array.isArray(assignments)) {
        outputDiv.innerHTML += `<p style="color:red;">No routing assignments found to plot.</p>`;
        return;
      }

      renderHeatmap(assignments, expertIds.length);
      analyzeRouting(assignments, expertIds.length);
    })
    .catch(err => {
      document.getElementById("output").innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
    });
});

// 🔥 Render Matrix Heatmap (if plugin available) or fallback to raw canvas grid
function renderHeatmap(assignments, numExperts) {
  const tokens = assignments.map(a => a.token_id);
  const expertLoads = Array.from({ length: tokens.length }, (_, i) =>
    Array.from({ length: numExperts }, (_, j) => ({ x: j, y: i, v: 0 }))
  );

  assignments.forEach(a => {
    const t = a.token_id;
    const e = a.expert;
    expertLoads[t][e].v = 1;
  });

  const flatData = expertLoads.flat();
  const canvas = document.getElementById("heatmapCanvas");
  const ctx = canvas.getContext("2d");

  if (window.routingHeatmap && typeof window.routingHeatmap.destroy === "function") {
    window.routingHeatmap.destroy();
  }

  if (matrixPluginLoaded()) {
    window.routingHeatmap = new Chart(ctx, {
      type: "matrix",
      data: {
        datasets: [{
          label: "Token → Expert Assignment",
          data: flatData,
          backgroundColor(ctx) {
            return ctx.raw.v === 1 ? "rgba(0, 123, 255, 0.8)" : "rgba(230, 230, 230, 0.15)";
          },
          width: () => 18,
          height: () => 18
        }]
      },
      options: {
        plugins: {
          tooltip: {
            callbacks: {
              title: ctx => `Token ${ctx[0].raw.y}`,
              label: ctx => `Expert ${ctx.raw.x}: ${ctx.raw.v ? "Assigned" : "Not used"}`
            }
          }
        },
        scales: {
          x: { title: { display: true, text: "Expert ID" }, ticks: { stepSize: 1 } },
          y: { title: { display: true, text: "Token ID" }, ticks: { stepSize: 1 } }
        }
      }
    });
  } else {
    // 🧱 Fallback: draw manually
    const cell = 18;
    canvas.width = numExperts * cell;
    canvas.height = tokens.length * cell;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#ddd";
    for (let i = 0; i <= numExperts; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cell, 0);
      ctx.lineTo(i * cell, canvas.height);
      ctx.stroke();
    }
    for (let j = 0; j <= tokens.length; j++) {
      ctx.beginPath();
      ctx.moveTo(0, j * cell);
      ctx.lineTo(canvas.width, j * cell);
      ctx.stroke();
    }
    flatData.forEach(d => {
      if (d.v === 1) {
        ctx.fillStyle = "rgba(0, 123, 255, 0.8)";
        ctx.fillRect(d.x * cell, d.y * cell, cell, cell);
      }
    });
  }
}

// 📊 Print analysis summary
function analyzeRouting(assignments, numExperts) {
  const expertCount = Array(numExperts).fill(0);
  assignments.forEach(a => expertCount[a.expert]++);

  const maxLoad = Math.max(...expertCount);
  const minLoad = Math.min(...expertCount);
  const avg = (expertCount.reduce((a, b) => a + b, 0) / numExperts).toFixed(2);

  const overloaded = expertCount.map((c, i) => c > avg * 1.5 ? i : null).filter(x => x !== null);
  const underused = expertCount.map((c, i) => c < avg * 0.5 ? i : null).filter(x => x !== null);

  const insights = `
    <h4>📊 Routing Analysis</h4>
    <ul>
      <li><strong>Max Load:</strong> ${maxLoad}</li>
      <li><strong>Min Load:</strong> ${minLoad}</li>
      <li><strong>Average Load:</strong> ${avg}</li>
      <li><strong>Overloaded Experts:</strong> ${overloaded.length > 0 ? overloaded.join(", ") : "None"}</li>
      <li><strong>Underused Experts:</strong> ${underused.length > 0 ? underused.join(", ") : "None"}</li>
    </ul>
  `;

  document.getElementById("output").innerHTML += insights;
}
