document.addEventListener("DOMContentLoaded", () => {
  const energyForm = document.getElementById("energyForm");
  if (!energyForm) return;

  energyForm.addEventListener("submit", function (e) {
    e.preventDefault();

    try {
      const jobGraph = JSON.parse(document.getElementById("jobGraph").value);
      const energyProfile = JSON.parse(document.getElementById("energyProfile").value);

      const useQuantumEl = document.getElementById("useQuantum");
      const useMLEl = document.getElementById("useML");
      const useHybridEl = document.getElementById("useHybrid");

      const useQuantum = useQuantumEl?.checked || false;
      const useML = useMLEl?.checked || false;
      const useHybrid = useHybridEl?.checked || false;

      const maxEnergy = Number(document.getElementById("maxEnergy")?.value || 0);
      const qIterations = Number(document.getElementById("qIterations")?.value || 0);
      const learningRate = Number(document.getElementById("learningRate")?.value || 0);
      const batchSize = Number(document.getElementById("batchSize")?.value || 0);

      let warning = "";
      if (maxEnergy < 1 || maxEnergy > 1000) {
        warning = "Max Energy Limit must be between 1 and 1000.";
      } else if (qIterations < 1 || qIterations > 100) {
        warning = "Quantum Iterations must be between 1 and 100.";
      } else if (learningRate < 0.0001 || learningRate > 1.0) {
        warning = "Learning Rate must be between 0.0001 and 1.0.";
      } else if (batchSize < 1 || batchSize > 1024) {
        warning = "Batch Size must be between 1 and 1024.";
      }

      if (warning) {
        const warningEl = document.getElementById("warning");
        if (warningEl) warningEl.textContent = warning;
        return;
      } else {
        const warningEl = document.getElementById("warning");
        if (warningEl) warningEl.textContent = "";
      }

      fetch("/q-energy/schedule", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_graph: jobGraph,
          energy_profile: energyProfile,
          use_quantum: useQuantum,
          use_ml: useML,
          use_qaoa: false,
          use_hybrid: useHybrid,
          max_energy_limit: maxEnergy,
          quantum_iterations: qIterations,
          learning_rate: learningRate,
          batch_size: batchSize
        })
      })
        .then(res => res.json())
        .then(data => {
          document.getElementById("energyOutput").textContent = JSON.stringify(data, null, 2);
        })
        .catch(err => {
          document.getElementById("energyOutput").textContent = `Error: ${err.message}`;
        });

    } catch (err) {
      document.getElementById("energyOutput").textContent = `Error parsing input: ${err.message}`;
    }
  });
});
