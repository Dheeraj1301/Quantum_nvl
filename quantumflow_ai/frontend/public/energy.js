document.getElementById("energyForm").addEventListener("submit", function (e) {
  e.preventDefault();
  const jobGraph = JSON.parse(document.getElementById("jobGraph").value);
  const energyProfile = JSON.parse(document.getElementById("energyProfile").value);
  const useQuantum = document.getElementById("useQuantum").checked;
  const useML = document.getElementById("useML").checked;
  const useHybrid = document.getElementById("useHybrid").checked;

  fetch("/q-energy/schedule", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_graph: jobGraph,
      energy_profile: energyProfile,
      use_quantum: useQuantum,
      use_ml: useML,
      use_qaoa: false,
      use_hybrid: useHybrid
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
