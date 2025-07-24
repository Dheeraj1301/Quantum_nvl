(function () {
  function getEl(id) {
    const el = document.getElementById(id);
    if (!el) console.warn(`Element with id "${id}" not found.`);
    return el;
  }

  function parseJSON(value, fallback = {}) {
    try {
      return JSON.parse(value);
    } catch (err) {
      console.warn("Invalid JSON. Using fallback.", err);
      return fallback;
    }
  }

  function init() {
    const form = getEl("energyForm");
    if (!form) return;

    const quantumEl = getEl("useQuantum");
    const mlEl = getEl("useML");

    // Auto-disable conflicting checkboxes
    if (quantumEl && mlEl) {
      quantumEl.addEventListener("change", () => {
        if (quantumEl.checked) mlEl.checked = false;
      });
      mlEl.addEventListener("change", () => {
        if (mlEl.checked) quantumEl.checked = false;
      });
    }

    form.addEventListener("submit", function (e) {
      e.preventDefault();

      const jobGraph = parseJSON(getEl("jobGraph")?.value || "{}");
      const energyProfile = parseJSON(getEl("energyProfile")?.value || "{}");

      const useQuantum = quantumEl?.checked ?? false;
      const useML = mlEl?.checked ?? false;
      const useHybrid = getEl("useHybrid")?.checked ?? false;
      const useMeta = getEl("useMeta")?.checked ?? false;
      const useGnn = getEl("useGnn")?.checked ?? false;

      const maxEnergy = Number(getEl("maxEnergy")?.value || 100);
      const qIterations = Number(getEl("qIterations")?.value || 10);
      const learningRate = Number(getEl("learningRate")?.value || 0.01);
      const batchSize = Number(getEl("batchSize")?.value || 32);

      let warning = "";
      if (isNaN(maxEnergy) || maxEnergy < 1 || maxEnergy > 1000) {
        warning = "Max Energy Limit must be between 1 and 1000.";
      } else if (isNaN(qIterations) || qIterations < 1 || qIterations > 100) {
        warning = "Quantum Iterations must be between 1 and 100.";
      } else if (isNaN(learningRate) || learningRate < 0.0001 || learningRate > 1.0) {
        warning = "Learning Rate must be between 0.0001 and 1.0.";
      } else if (isNaN(batchSize) || batchSize < 1 || batchSize > 1024) {
        warning = "Batch Size must be between 1 and 1024.";
      }

      const warningEl = getEl("warning");
      if (warning) {
        if (warningEl) warningEl.textContent = warning;
        return;
      } else if (warningEl) {
        warningEl.textContent = "";
      }

      const payload = {
        job_graph: jobGraph,
        energy_profile: energyProfile,
        use_quantum: useQuantum,
        use_ml: useML,
        use_qaoa: false,
        use_hybrid: useHybrid,
        use_meta: useMeta,
        use_gnn: useGnn,
        max_energy_limit: maxEnergy,
        quantum_iterations: qIterations,
        learning_rate: learningRate,
        batch_size: batchSize
      };

      fetch("/q-energy/schedule", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
        .then(res => res.json())
        .then(data => {
          const outEl = getEl("energyOutput");
          if (outEl) outEl.textContent = JSON.stringify(data, null, 2);
        })
        .catch(err => {
          const outEl = getEl("energyOutput");
          if (outEl) outEl.textContent = `Error: ${err.message}`;
        });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
