document.addEventListener("DOMContentLoaded", () => {
  const energyForm = document.getElementById("energyForm");
  if (!energyForm) return;

  energyForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const jobGraph = JSON.parse(document.getElementById("jobGraph").value);
    const energyProfile = JSON.parse(document.getElementById("energyProfile").value);
    const useQuantum = document.getElementById("useQuantum")?.checked || false;
    const useML = document.getElementById("useML")?.checked || false;
    const useHybrid = document.getElementById("useHybrid")?.checked || false;
    const maxEnergy = Number(document.getElementById("maxEnergy").value);
    const qIterations = Number(document.getElementById("qIterations").value);
    const learningRate = Number(document.getElementById("learningRate").value);
    const batchSize = Number(document.getElementById("batchSize").value);

  let warning = "";
  if (maxEnergy < 1 || maxEnergy > 1000) {
    warning = "Max Energy Limit must be between 1 and 1000.";
  } else if (qIterations < 1 || qIterations > 100) {
    warning = "Quantum Iterations must be between 1 and 100.";
  } else if (learningRate < 0.0001 || learningRate > 1) {
    warning = "Learning Rate must be between 0.0001 and 1.0.";
  } else if (batchSize < 1 || batchSize > 1024) {
    warning = "Batch Size must be between 1 and 1024.";
  }

  if (warning) {
    document.getElementById("warning").textContent = warning;
    return;
  } else {
    document.getElementById("warning").textContent = "";
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
});
});
